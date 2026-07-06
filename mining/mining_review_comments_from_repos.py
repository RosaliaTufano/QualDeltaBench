from __future__ import annotations

import base64
import json
import os
import random
import time
from typing import Any

import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Select the language to mine. This is the only language-specific setting to change.
# Options:
#   LANGUAGE = "java"
#   LANGUAGE = "python"
LANGUAGE = "python"

LANGUAGE_CONFIG = {
    "java": {
        "repos_csv_stem": "500_random_java_repos",
        "file_extension": ".java",
    },
    "python": {
        "repos_csv_stem": "500_random_python_repos",
        "file_extension": ".py",
    },
}

if LANGUAGE not in LANGUAGE_CONFIG:
    valid_languages = ", ".join(sorted(LANGUAGE_CONFIG))
    raise ValueError(f"Unsupported LANGUAGE={LANGUAGE!r}. Use one of: {valid_languages}.")

REPOS_CSV_STEM = LANGUAGE_CONFIG[LANGUAGE]["repos_csv_stem"]
TARGET_FILE_EXTENSION = LANGUAGE_CONFIG[LANGUAGE]["file_extension"]

GITHUB_TOKEN_ENV_VAR = "GITHUB_TOKEN"
MAX_RETRIES = 20
REQUEST_TIMEOUT = (5, 60)
MAX_FILE_BYTES = 200_000
GITHUB_API_VERSION = "2022-11-28"

OUT_DIR = f"pr_review_comments_{REPOS_CSV_STEM}"
STATE_DIR = f"state_{REPOS_CSV_STEM}"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

GITHUB_TOKEN = os.getenv(GITHUB_TOKEN_ENV_VAR)
if not GITHUB_TOKEN:
    raise RuntimeError(
        f"Missing GitHub token. Set the {GITHUB_TOKEN_ENV_VAR} environment "
        "variable before running this script."
    )

repos_df = pd.read_csv(f"{REPOS_CSV_STEM}.csv")
REPOS = repos_df["name"].to_list()
print(f"Number of repositories: {len(REPOS)}")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
)

FILE_CACHE: dict[tuple[str, str, str], str | None] = {}


# -----------------------------------------------------------------------------
# GitHub API helpers
# -----------------------------------------------------------------------------

def safe_filename(repo: str) -> str:
    return repo.replace("/", "__")


def request_with_retry(
    url: str,
    params: dict[str, Any] | None = None,
    max_retries: int = MAX_RETRIES,
    timeout: tuple[int, int] = REQUEST_TIMEOUT,
) -> requests.Response | None:
    for attempt in range(max_retries):
        try:
            response = SESSION.get(url, params=params, timeout=timeout)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            backoff = min(60, 2**attempt) + random.random()
            print(
                f"[NET] {type(exc).__name__} "
                f"(attempt {attempt + 1}/{max_retries}); "
                f"sleeping {backoff:.1f}s: {url}"
            )
            time.sleep(backoff)
            continue

        if response.status_code == 404:
            return None

        response_text = response.text.lower()

        if response.status_code in (403, 429) and (
            "secondary rate limit" in response_text
            or "abuse detection" in response_text
        ):
            backoff = min(120, 5 * (attempt + 1)) + random.random()
            print(f"[SECONDARY RATE LIMIT] sleeping {backoff:.1f}s: {url}")
            time.sleep(backoff)
            continue

        if response.status_code == 403 and "rate limit" in response_text:
            reset = response.headers.get("X-RateLimit-Reset")
            if reset:
                sleep_for = max(0, int(reset) - int(time.time())) + 2
                print(f"[RATE LIMIT] sleeping {sleep_for}s")
                time.sleep(sleep_for)
                continue

        if response.status_code in (500, 502, 503, 504):
            backoff = min(60, 2**attempt) + random.random()
            print(f"[{response.status_code}] retrying in {backoff:.1f}s: {url}")
            time.sleep(backoff)
            continue

        if response.status_code == 429:
            backoff = min(180, 10 * (attempt + 1)) + random.random()
            print(f"[429 TOO MANY REQUESTS] sleeping {backoff:.1f}s")
            time.sleep(backoff)
            continue

        response.raise_for_status()
        return response

    raise RuntimeError(f"Too many failed retries: {url}")


def paginate(url: str, params: dict[str, Any] | None = None):
    page = 1
    while True:
        page_params = dict(params or {})
        page_params.update({"per_page": 100, "page": page})
        response = request_with_retry(url, params=page_params)

        if response is None:
            return

        items = response.json()
        if not items:
            break

        yield from items
        page += 1


def fetch_all_prs(repo: str) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{repo}/pulls"
    pull_requests = []

    for pr in paginate(url, params={"state": "all"}):
        reviewers = [
            user.get("login")
            for user in (pr.get("requested_reviewers") or [])
            if user.get("login")
        ]
        teams = [
            team.get("slug")
            for team in (pr.get("requested_teams") or [])
            if team.get("slug")
        ]

        state = pr.get("state")
        merged_at = pr.get("merged_at")
        if state == "closed" and merged_at:
            pr_status = "merged"
        elif state == "closed":
            pr_status = "closed"
        else:
            pr_status = "open"

        pull_requests.append(
            {
                "pr_number": pr["number"],
                "pr_html_url": pr.get("html_url"),
                "pr_state": state,
                "merged_at": merged_at,
                "closed_at": pr.get("closed_at"),
                "created_at": pr.get("created_at"),
                "pr_status": pr_status,
                "requested_reviewers": ";".join(reviewers),
                "requested_teams": ";".join(teams),
            }
        )

    return pull_requests


def fetch_pr_commits(repo: str, pr_number: int) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/commits"
    commits = []

    for commit in paginate(url):
        commit_obj = commit.get("commit", {}) or {}
        author_obj = commit_obj.get("author") or {}
        committer_obj = commit_obj.get("committer") or {}
        github_author = commit.get("author") or {}

        commits.append(
            {
                "repo": repo,
                "pr_number": pr_number,
                "sha": commit.get("sha"),
                "commit_html_url": commit.get("html_url"),
                "commit_author_login": github_author.get("login"),
                "message": commit_obj.get("message"),
                "author_name": author_obj.get("name"),
                "author_email": author_obj.get("email"),
                "author_date": author_obj.get("date"),
                "committer_name": committer_obj.get("name"),
                "committer_email": committer_obj.get("email"),
                "committer_date": committer_obj.get("date"),
            }
        )

    return commits


def fetch_pr_review_comments(repo: str, pr_number: int) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
    comments = []

    for comment in paginate(url):
        comments.append(
            {
                "repo": repo,
                "pr_number": pr_number,
                "review_comment_id": comment.get("id"),
                "review_comment_html_url": comment.get("html_url"),
                "user_login": (comment.get("user") or {}).get("login"),
                "created_at": comment.get("created_at"),
                "updated_at": comment.get("updated_at"),
                "body": comment.get("body"),
                "role_from_api": comment.get("author_association"),
                "diff_hunk": comment.get("diff_hunk"),
                "commit_id": comment.get("original_commit_id"),
                "path": comment.get("path"),
                "position": comment.get("position"),
                "line": comment.get("line"),
                "original_line": comment.get("original_line"),
                "side": comment.get("side"),
                "start_line": comment.get("start_line"),
                "original_start_line": comment.get("original_start_line"),
                "start_side": comment.get("start_side"),
                "in_reply_to_id": comment.get("in_reply_to_id"),
            }
        )

    return comments


def fetch_file_at_commit(
    repo: str,
    sha: str,
    path: str,
    max_bytes: int = MAX_FILE_BYTES,
) -> str | None:
    if not repo or not sha or not path:
        return None

    cache_key = (repo, sha, path)
    if cache_key in FILE_CACHE:
        return FILE_CACHE[cache_key]

    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        response = request_with_retry(url, params={"ref": sha})
    except Exception:
        FILE_CACHE[cache_key] = None
        return None

    if response is None:
        FILE_CACHE[cache_key] = None
        return None

    try:
        data = response.json()
    except Exception:
        FILE_CACHE[cache_key] = None
        return None

    if isinstance(data, list):
        FILE_CACHE[cache_key] = None
        return None

    text = None
    if data.get("encoding") == "base64" and "content" in data:
        try:
            raw = base64.b64decode(data["content"])
            text = raw[:max_bytes].decode("utf-8", errors="ignore")
        except Exception:
            text = None

    FILE_CACHE[cache_key] = text
    return text


# -----------------------------------------------------------------------------
# Local state and CSV helpers
# -----------------------------------------------------------------------------

def load_progress(repo: str) -> int:
    path = os.path.join(STATE_DIR, f"{safe_filename(repo)}.json")
    if not os.path.exists(path):
        return 0

    with open(path, "r", encoding="utf-8") as file:
        return int(json.load(file).get("last_pr_number", 0))


def save_progress(repo: str, last_pr_number: int) -> None:
    path = os.path.join(STATE_DIR, f"{safe_filename(repo)}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"last_pr_number": last_pr_number}, file)


def append_rows_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    dataframe = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    dataframe.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8")


# -----------------------------------------------------------------------------
# Review-comment filtering
# -----------------------------------------------------------------------------

def check_commented_line(
    content_file: str | None,
    diff_hunk: str | None,
    line_to_check: int | None,
) -> tuple[str, str, bool]:
    if content_file is None or diff_hunk is None or line_to_check is None:
        raise ValueError("Missing file content, diff hunk, or line number.")

    file_lines = content_file.split("\n")
    if line_to_check < 1 or line_to_check > len(file_lines):
        raise ValueError("Line number is outside the fetched file content.")

    line_to_check_file = file_lines[line_to_check - 1]
    diff_hunk_lines = diff_hunk.split("\n")
    if not diff_hunk_lines or "+" not in diff_hunk_lines[0]:
        raise ValueError("Diff hunk header does not contain target-line metadata.")

    current_line_number = int(diff_hunk_lines[0].split("+")[1].split(",")[0])
    relevant_lines = []
    relevant_line_numbers = []

    for line in diff_hunk_lines[1:]:
        if not line:
            continue
        if line[0] == "-":
            continue

        relevant_lines.append(line[1:])
        relevant_line_numbers.append(current_line_number)
        current_line_number += 1

    if line_to_check not in relevant_line_numbers:
        raise ValueError("Commented line is not present in the diff hunk.")

    line_to_check_diff_hunk = relevant_lines[relevant_line_numbers.index(line_to_check)]
    is_valid = line_to_check_file == line_to_check_diff_hunk
    return line_to_check_file, line_to_check_diff_hunk, is_valid


def is_test_file(path: str) -> bool:
    return "test" in os.path.basename(path).lower()


def add_pr_metadata(
    rows: list[dict[str, Any]],
    pr_html_url: str | None,
    requested_reviewers: str,
    requested_teams: str,
    pr_status: str | None = None,
    pr_state: str | None = None,
    merged_at: str | None = None,
) -> None:
    for row in rows:
        row["pr_html_url"] = pr_html_url
        row["requested_reviewers"] = requested_reviewers
        row["requested_teams"] = requested_teams
        if pr_status is not None:
            row["pr_status"] = pr_status
        if pr_state is not None:
            row["pr_state"] = pr_state
        if merged_at is not None:
            row["merged_at"] = merged_at


def empty_summary_row(repo: str, pr_number: int) -> dict[str, Any]:
    return {
        "repo": repo,
        "pr_number": pr_number,
        "pr_html_url": None,
        "pr_status": None,
        "pr_state": None,
        "merged_at": None,
        "closed_at": None,
        "created_at": None,
        "requested_reviewers": None,
        "requested_teams": None,
        "n_commits": None,
        "n_review_comments_total": None,
        "n_review_comments_kept": None,
        "n_review_comments_to_non_target_file": None,
        "n_review_comments_to_test_file": None,
        "n_review_comments_discarded_left": None,
        "n_review_comments_discarded_no_code_line": None,
        "n_review_comments_discarded_diff_mismatch": None,
        "error": "commits_endpoint_404_or_unavailable",
    }


def process_repo(repo: str) -> None:
    review_comments_path = os.path.join(OUT_DIR, f"{safe_filename(repo)}_pr_review_comments.csv")
    pr_summary_path = os.path.join(OUT_DIR, f"{safe_filename(repo)}_pr_summary.csv")

    last_done = load_progress(repo)
    print(f"\n[REPO] {repo} (resuming from PR > {last_done})")

    pull_requests = fetch_all_prs(repo)
    pull_requests.sort(key=lambda item: item["pr_number"])

    for pr in pull_requests:
        pr_number = pr["pr_number"]
        if pr_number <= last_done:
            continue

        pr_html_url = pr.get("pr_html_url")
        requested_reviewers_str = pr.get("requested_reviewers") or ""
        requested_reviewers = {
            reviewer for reviewer in requested_reviewers_str.split(";") if reviewer
        }
        requested_teams_str = pr.get("requested_teams") or ""
        pr_status = pr.get("pr_status")
        pr_state = pr.get("pr_state")
        merged_at = pr.get("merged_at")

        commits = fetch_pr_commits(repo, pr_number)
        if not commits:
            append_rows_csv(pr_summary_path, [empty_summary_row(repo, pr_number)])
            save_progress(repo, pr_number)
            print(
                f"[WARN] {repo} - PR #{pr_number}: commits endpoint unavailable; "
                "empty summary saved."
            )
            continue

        sha_to_author_login = {
            row["sha"]: row.get("commit_author_login")
            for row in commits
            if row.get("sha")
        }

        review_comments = fetch_pr_review_comments(repo, pr_number)
        filtered_review_comments = []

        non_target_file = 0
        test_file = 0
        comment_left = 0
        comment_no_code_line = 0
        comment_on_different_commit = 0

        for row in review_comments:
            path = row.get("path") or ""
            if not path.endswith(TARGET_FILE_EXTENSION):
                non_target_file += 1
                continue

            if is_test_file(path):
                test_file += 1
                continue

            if row.get("line") is None and row.get("start_line") is None:
                comment_no_code_line += 1
                continue

            if row.get("side") == "LEFT":
                comment_left += 1
                continue

            add_pr_metadata(
                [row],
                pr_html_url,
                requested_reviewers_str,
                requested_teams_str,
                pr_status,
                pr_state,
                merged_at,
            )

            commenter = row.get("user_login")
            commit_sha = row.get("commit_id")
            commit_author = sha_to_author_login.get(commit_sha)
            row["commit_author_login"] = commit_author

            if commenter and commit_author and commenter == commit_author:
                row["role"] = "contributor"
            elif commenter and commenter in requested_reviewers:
                row["role"] = "assigned_reviewer"
            else:
                row["role"] = "reviewer"

            if not commit_sha or not path:
                comment_on_different_commit += 1
                continue

            row["file_content_at_comment_commit"] = fetch_file_at_commit(
                repo, commit_sha, path
            )
            content_file = row.get("file_content_at_comment_commit")
            diff_hunk = row.get("diff_hunk")
            line_to_check = row.get("original_line") or row.get("line")

            try:
                file_line, diff_line, is_valid = check_commented_line(
                    content_file, diff_hunk, line_to_check
                )
            except Exception:
                comment_on_different_commit += 1
                continue

            if not is_valid:
                comment_on_different_commit += 1
                continue

            row["file_line"] = file_line
            row["diff_line"] = diff_line
            row["is_comment_valid"] = is_valid
            filtered_review_comments.append(row)

        append_rows_csv(review_comments_path, filtered_review_comments)

        summary_row = {
            "repo": repo,
            "pr_number": pr_number,
            "pr_html_url": pr_html_url,
            "pr_status": pr_status,
            "pr_state": pr_state,
            "merged_at": merged_at,
            "n_commits": len(commits),
            "n_review_comments_total": len(review_comments),
            "n_review_comments_kept": len(filtered_review_comments),
            "n_review_comments_to_non_target_file": non_target_file,
            "n_review_comments_to_test_file": test_file,
            "n_review_comments_discarded_left": comment_left,
            "n_review_comments_discarded_no_code_line": comment_no_code_line,
            "n_review_comments_discarded_diff_mismatch": comment_on_different_commit,
        }
        append_rows_csv(pr_summary_path, [summary_row])
        save_progress(repo, pr_number)

        print(
            f"[OK] {repo} - PR #{pr_number}: "
            f"{len(commits)} commits fetched for validation, "
            f"{len(review_comments)} review comments, "
            f"{len(filtered_review_comments)} filtered review comments"
        )


def main() -> None:
    for repo in REPOS:
        process_repo(repo)


if __name__ == "__main__":
    main()
