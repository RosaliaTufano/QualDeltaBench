from __future__ import annotations

import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Options:
#   LANGUAGE = "java"
#   LANGUAGE = "python"
LANGUAGE = "python"

LANGUAGE_CONFIG = {
    "java": {
        "repos_csv_stem": "500_random_java_repos",
        "target_file_extension": ".java",
        "single_input_csv": "comments_java.csv",
    },
    "python": {
        "repos_csv_stem": "500_random_python_repos",
        "target_file_extension": ".py",
        "single_input_csv": "comments_python.csv",
    },
}

if LANGUAGE not in LANGUAGE_CONFIG:
    valid_languages = ", ".join(sorted(LANGUAGE_CONFIG))
    raise ValueError(f"Unsupported LANGUAGE={LANGUAGE!r}. Use one of: {valid_languages}.")

REPOS_CSV_STEM = LANGUAGE_CONFIG[LANGUAGE]["repos_csv_stem"]
TARGET_FILE_EXTENSION = LANGUAGE_CONFIG[LANGUAGE]["target_file_extension"]

# Input option 1: use one CSV file containing filtered review comments.
# Set to None to read all *_pr_review_comments.csv files from INPUT_COMMENTS_DIR.
INPUT_COMMENTS_CSV_PATH: str | None = None
# INPUT_COMMENTS_CSV_PATH = LANGUAGE_CONFIG[LANGUAGE]["single_input_csv"]

# Input option 2: use the output directory produced by mining_review_comments_only.py.
INPUT_COMMENTS_DIR = f"pr_review_comments_{REPOS_CSV_STEM}"
INPUT_COMMENTS_GLOB = "*_pr_review_comments.csv"

# Optional input: if PR commit CSVs from the full mining script are available,
# they will be used. If they are missing, the script fetches PR commits from the
# GitHub API instead.
PR_COMMITS_DIRS = [f"pr_commits_{REPOS_CSV_STEM}"]

OUTPUT_DIR = f"commit_after_{REPOS_CSV_STEM}"
OUTPUT_JSONL_PATH = os.path.join(
    OUTPUT_DIR, f"filtered_comments_with_commit_after_{LANGUAGE}.jsonl"
)
CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR, f"filtered_comments_with_commit_after_{LANGUAGE}.checkpoint.json"
)
RELEVANT_COMMENTS_CSV_PATH = os.path.join(
    OUTPUT_DIR, f"relevant_comments_{LANGUAGE}.csv"
)

GITHUB_TOKEN_ENV_VAR = "GITHUB_TOKEN"
GITHUB_API_VERSION = "2022-11-28"
REQUEST_TIMEOUT = (5, 60)
MAX_RETRIES = 20
CHECKPOINT_EVERY = 50


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

GITHUB_TOKEN = os.getenv(GITHUB_TOKEN_ENV_VAR)
if not GITHUB_TOKEN:
    raise RuntimeError(
        f"Missing GitHub token. Set the {GITHUB_TOKEN_ENV_VAR} environment "
        "variable before running this script."
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
)

RELEVANT_CSV_FIELDS = [
    "row_id",
    "id",
    "repo",
    "pr_number",
    "review_comment_id",
    "review_comment_html_url",
    "commit_id",
    "commit_after",
    "path",
    "line",
    "original_line",
    "body",
    "code_before",
    "code_after",
]

HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


# -----------------------------------------------------------------------------
# File and CSV helpers
# -----------------------------------------------------------------------------

def atomic_write_json(path: str, obj: dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def append_jsonl_record(path: str, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_csv_record(path: str, record: dict[str, Any], fieldnames: list[str]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {
            "next_row_id": 0,
            "last_written_row_id": None,
            "processed_rows_total": 0,
            "commit_after_found_total": 0,
            "commit_after_not_found_total": 0,
            "file_after_found_total": 0,
            "modifies_commented_line_true_total": 0,
            "relevant_comments_total": 0,
            "updated_at_epoch": None,
        }

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def update_checkpoint(
    path: str,
    next_row_id: int,
    last_written_row_id: int,
    stats: dict[str, int],
) -> None:
    payload = {
        "next_row_id": next_row_id,
        "last_written_row_id": last_written_row_id,
        "processed_rows_total": stats["processed_rows_total"],
        "commit_after_found_total": stats["commit_after_found_total"],
        "commit_after_not_found_total": stats["commit_after_not_found_total"],
        "file_after_found_total": stats["file_after_found_total"],
        "modifies_commented_line_true_total": stats[
            "modifies_commented_line_true_total"
        ],
        "relevant_comments_total": stats["relevant_comments_total"],
        "updated_at_epoch": time.time(),
    }
    atomic_write_json(path, payload)


def read_comments_input() -> pd.DataFrame:
    if INPUT_COMMENTS_CSV_PATH:
        path = Path(INPUT_COMMENTS_CSV_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        return pd.read_csv(path).reset_index(drop=True)

    input_dir = Path(INPUT_COMMENTS_DIR)
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}. Either create it with the "
            "review-comment mining script or set INPUT_COMMENTS_CSV_PATH."
        )

    paths = sorted(input_dir.glob(INPUT_COMMENTS_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {INPUT_COMMENTS_GLOB!r} found in {input_dir}."
        )

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def normalize_pr_number(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def load_pr_commits_index(path: str) -> dict[str, list[str]]:
    pr_to_commits: dict[str, list[str]] = {}

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")

        required_columns = {"pr_number", "sha"}
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {path}: {sorted(missing_columns)}"
            )

        for row in reader:
            pr_number = normalize_pr_number(row.get("pr_number"))
            sha = (row.get("sha") or "").strip()
            if pr_number and sha:
                pr_to_commits.setdefault(pr_number, []).append(sha)

    return pr_to_commits


def build_repo_to_pr_commits_file_map(pr_commit_dirs: list[str]) -> dict[str, str]:
    repo_to_file: dict[str, str] = {}

    for base_dir in pr_commit_dirs:
        if not os.path.isdir(base_dir):
            print(f"[WARNING] optional PR commits directory not found: {base_dir}")
            continue

        for filename in os.listdir(base_dir):
            if not filename.endswith("_pr_commits.csv"):
                continue

            repo_key = filename.rsplit("_pr_commits.csv", 1)[0]
            repo_to_file[repo_key] = os.path.join(base_dir, filename)

    return repo_to_file


# -----------------------------------------------------------------------------
# Data conversion helpers
# -----------------------------------------------------------------------------

def is_missing(value: Any) -> bool:
    return value is None or pd.isna(value) or value == ""


def to_int_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def get_commented_line(row: pd.Series) -> int | None:
    original_line = row.get("original_line")
    line = row.get("line")

    if not is_missing(original_line):
        return to_int_or_none(original_line)

    return to_int_or_none(line)


def has_required_fields(row: pd.Series) -> tuple[bool, str | None]:
    required_fields = ["repo", "pr_number", "commit_id", "path"]

    for field in required_fields:
        value = row.get(field)
        if is_missing(value):
            return False, f"missing_required_field:{field}"

    return True, None


def sanitize_for_json(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def sanitize_for_csv(value: Any) -> Any:
    if value is None or pd.isna(value):
        return ""
    return value


def print_stats(prefix: str, stats: dict[str, int]) -> None:
    print(
        f"{prefix} "
        f"processed={stats['processed_rows_total']} | "
        f"commit_after_found={stats['commit_after_found_total']} | "
        f"commit_after_not_found={stats['commit_after_not_found_total']} | "
        f"file_after_found={stats['file_after_found_total']} | "
        f"modifies_commented_line_true="
        f"{stats['modifies_commented_line_true_total']} | "
        f"relevant_comments={stats['relevant_comments_total']}"
    )


def increment_stats(
    stats: dict[str, int], result: dict[str, Any], is_relevant: bool
) -> None:
    stats["processed_rows_total"] += 1

    if result.get("commit_after") != "-":
        stats["commit_after_found_total"] += 1

    if result.get("processing_error") == "commit_after_not_found":
        stats["commit_after_not_found_total"] += 1

    if result.get("file_after") != "-":
        stats["file_after_found_total"] += 1

    if result.get("modifies_commented_line") is True:
        stats["modifies_commented_line_true_total"] += 1

    if is_relevant:
        stats["relevant_comments_total"] += 1


# -----------------------------------------------------------------------------
# GitHub API helpers
# -----------------------------------------------------------------------------

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
                f"(attempt {attempt + 1}/{max_retries}); sleeping {backoff:.1f}s"
            )
            time.sleep(backoff)
            continue

        if response.status_code == 404:
            return None

        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        if remaining and remaining.isdigit() and int(remaining) == 0:
            sleep_for = 60
            if reset and reset.isdigit():
                sleep_for = max(0, int(reset) - int(time.time())) + 2
            print(f"[RATE LIMIT] sleeping {sleep_for}s")
            time.sleep(sleep_for)
            continue

        response_text = (response.text or "").lower()
        if response.status_code in (403, 429) and (
            "secondary rate limit" in response_text
            or "abuse detection" in response_text
            or "you have exceeded a secondary rate limit" in response_text
        ):
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                backoff = int(retry_after) + 1
            else:
                backoff = min(180, 10 * (attempt + 1)) + random.random()
            print(f"[SECONDARY RATE LIMIT] sleeping {backoff:.1f}s")
            time.sleep(backoff)
            continue

        if response.status_code in (500, 502, 503, 504):
            backoff = min(60, 2**attempt) + random.random()
            print(f"[{response.status_code}] retrying in {backoff:.1f}s")
            time.sleep(backoff)
            continue

        response.raise_for_status()
        return response

    raise RuntimeError(f"Too many failed retries: {url}")


def paginate(url: str, params: dict[str, Any] | None = None):
    page = 1
    while True:
        request_params = dict(params or {})
        request_params.update({"per_page": 100, "page": page})
        response = request_with_retry(url, params=request_params)
        if response is None:
            return

        data = response.json()
        if not data:
            return

        for item in data:
            yield item

        if len(data) < 100:
            return

        page += 1


def fetch_pr_commits(repo: str, pr_number: str | int) -> list[str] | None:
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/commits"
    commits = []

    for item in paginate(url):
        sha = item.get("sha")
        if sha:
            commits.append(sha)

    return commits


def fetch_commit_files_and_patches(repo: str, sha: str) -> list[dict[str, Any]] | None:
    url = f"https://api.github.com/repos/{repo}/commits/{sha}"
    response = request_with_retry(url, timeout=REQUEST_TIMEOUT)
    if response is None:
        return None

    data = response.json()
    files = []
    for file_data in data.get("files", []) or []:
        files.append(
            {
                "repo": repo,
                "sha": sha,
                "filename": file_data.get("filename"),
                "status": file_data.get("status"),
                "additions": file_data.get("additions"),
                "deletions": file_data.get("deletions"),
                "changes": file_data.get("changes"),
                "patch": file_data.get("patch"),
            }
        )

    return files


def get_commit_files_cached(
    repo: str,
    sha: str,
    commit_api_cache: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    cache_key = (repo, sha)
    if cache_key in commit_api_cache:
        return commit_api_cache[cache_key]

    files = fetch_commit_files_and_patches(repo, sha)
    if files is None:
        result = [
            {
                "repo": repo,
                "sha": sha,
                "filename": None,
                "status": None,
                "additions": None,
                "deletions": None,
                "changes": None,
                "patch": None,
                "error": "not_found",
            }
        ]
        commit_api_cache[cache_key] = result
        return result

    for row in files:
        row["error"] = None

    commit_api_cache[cache_key] = files
    return files


# -----------------------------------------------------------------------------
# Diff parsing
# -----------------------------------------------------------------------------

def parse_hunk_new_file_line_ranges(diff: str | None) -> list[int]:
    if not isinstance(diff, str) or not diff.strip():
        return []

    touched_lines: list[int] = []
    for line in diff.splitlines():
        if not line.startswith("@@ "):
            continue

        match = HUNK_HEADER_RE.match(line)
        if not match:
            continue

        new_start = int(match.group(3))
        new_len = int(match.group(4)) if match.group(4) is not None else 1
        if new_len <= 0:
            continue

        new_end = new_start + new_len - 1
        touched_lines.extend(range(new_start, new_end + 1))

    return touched_lines


def extract_changed_code_for_commented_hunk(
    diff: str | None,
    commented_line: int | None,
) -> tuple[str | None, str | None]:
    if not isinstance(diff, str) or not diff.strip() or commented_line is None:
        return None, None

    lines = diff.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.startswith("@@ "):
            index += 1
            continue

        match = HUNK_HEADER_RE.match(line)
        if not match:
            index += 1
            continue

        new_start = int(match.group(3))
        new_len = int(match.group(4)) if match.group(4) is not None else 1
        new_end = new_start + new_len - 1 if new_len > 0 else new_start - 1

        index += 1
        hunk_lines = []
        while index < len(lines) and not lines[index].startswith("@@ "):
            hunk_lines.append(lines[index])
            index += 1

        if not (new_len > 0 and new_start <= commented_line <= new_end):
            continue

        before_lines = []
        after_lines = []
        for hunk_line in hunk_lines:
            if hunk_line.startswith("-") and not hunk_line.startswith("---"):
                before_lines.append(hunk_line[1:])
            elif hunk_line.startswith("+") and not hunk_line.startswith("+++"):
                after_lines.append(hunk_line[1:])

        return "\n".join(before_lines), "\n".join(after_lines)

    return None, None


# -----------------------------------------------------------------------------
# Commit lookup and row processing
# -----------------------------------------------------------------------------

def get_next_commit_after(relevant_commits: list[str], target_commit: str) -> str | None:
    found_target = False

    for sha in relevant_commits:
        if sha == target_commit:
            found_target = True
            continue

        if found_target:
            return sha

    return None


def get_pr_commits_for_row(
    repo: str,
    pr_number: str,
    repo_to_pr_file: dict[str, str],
    pr_file_cache: dict[str, dict[str, list[str]] | None],
    pr_api_cache: dict[tuple[str, str], list[str] | None],
) -> tuple[list[str] | None, str | None]:
    repo_key = repo.replace("/", "__")
    pr_number_normalized = normalize_pr_number(pr_number)
    if pr_number_normalized is None:
        return None, "invalid_pr_number"

    pr_info_path = repo_to_pr_file.get(repo_key)
    if pr_info_path is not None:
        if pr_info_path not in pr_file_cache:
            try:
                pr_file_cache[pr_info_path] = load_pr_commits_index(pr_info_path)
            except Exception as exc:
                pr_file_cache[pr_info_path] = None
                print(
                    f"[WARNING] failed to read PR commits file {pr_info_path}: "
                    f"{type(exc).__name__}: {exc}"
                )

        pr_index = pr_file_cache.get(pr_info_path)
        if pr_index is not None:
            commits = pr_index.get(pr_number_normalized, [])
            if commits:
                return commits, None

    cache_key = (repo, pr_number_normalized)
    if cache_key not in pr_api_cache:
        pr_api_cache[cache_key] = fetch_pr_commits(repo, pr_number_normalized)

    commits = pr_api_cache[cache_key]
    if commits:
        return commits, None

    if pr_info_path is None:
        return None, "pr_commits_not_found_in_file_or_api"
    return None, "pr_not_found_in_pr_commits_file_or_api"


def empty_result(error: str | None = None) -> dict[str, Any]:
    return {
        "commit_after": "-",
        "file_after": "-",
        "diff_after": "-",
        "modifies_commented_line": "-",
        "processing_error": error,
        "code_before": None,
        "code_after": None,
    }


def process_comment_row(
    row: pd.Series,
    repo_to_pr_file: dict[str, str],
    pr_file_cache: dict[str, dict[str, list[str]] | None],
    pr_api_cache: dict[tuple[str, str], list[str] | None],
    commit_api_cache: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    ok, error_msg = has_required_fields(row)
    if not ok:
        return empty_result(error_msg)

    repo = str(row.get("repo")).strip()
    pr_number = str(row.get("pr_number")).strip()
    commit_id = str(row.get("commit_id")).strip()
    file_path = str(row.get("path")).strip()

    if TARGET_FILE_EXTENSION and not file_path.endswith(TARGET_FILE_EXTENSION):
        return empty_result("non_target_file_extension")

    relevant_commits, error = get_pr_commits_for_row(
        repo=repo,
        pr_number=pr_number,
        repo_to_pr_file=repo_to_pr_file,
        pr_file_cache=pr_file_cache,
        pr_api_cache=pr_api_cache,
    )
    if error or not relevant_commits:
        return empty_result(error or "pr_commits_not_found")

    next_commit = get_next_commit_after(relevant_commits, commit_id)
    if next_commit is None:
        return empty_result("commit_after_not_found")

    result = empty_result()
    result["commit_after"] = next_commit

    files = get_commit_files_cached(
        repo=repo,
        sha=next_commit,
        commit_api_cache=commit_api_cache,
    )

    matched_file = None
    for file_row in files:
        if file_row.get("filename") == file_path:
            matched_file = file_row
            break

    if matched_file is None:
        result["processing_error"] = "file_not_found_in_commit_after"
        return result

    patch = matched_file.get("patch")
    result["file_after"] = matched_file.get("filename") or "-"
    result["diff_after"] = patch if patch is not None else "-"

    if patch is None:
        result["processing_error"] = "patch_not_available"
        return result

    modified_lines = parse_hunk_new_file_line_ranges(patch)
    if not modified_lines:
        result["processing_error"] = "no_hunks_in_patch"
        return result

    commented_line = get_commented_line(row)
    if commented_line is None:
        result["processing_error"] = "invalid_commented_line"
        return result

    result["modifies_commented_line"] = commented_line in modified_lines

    if result["modifies_commented_line"] is True:
        code_before, code_after = extract_changed_code_for_commented_hunk(
            diff=patch,
            commented_line=commented_line,
        )
        result["code_before"] = code_before
        result["code_after"] = code_after

    return result


# -----------------------------------------------------------------------------
# Output records
# -----------------------------------------------------------------------------

def build_output_record(row_id: int, row: pd.Series, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "id": sanitize_for_json(row.get("id")),
        "repo": sanitize_for_json(row.get("repo")),
        "pr_number": sanitize_for_json(row.get("pr_number")),
        "review_comment_id": sanitize_for_json(row.get("review_comment_id")),
        "review_comment_html_url": sanitize_for_json(
            row.get("review_comment_html_url")
        ),
        "commit_id": sanitize_for_json(row.get("commit_id")),
        "path": sanitize_for_json(row.get("path")),
        "line": sanitize_for_json(row.get("line")),
        "original_line": sanitize_for_json(row.get("original_line")),
        "body": sanitize_for_json(row.get("body")),
        "commit_after": result.get("commit_after"),
        "file_after": result.get("file_after"),
        "diff_after": result.get("diff_after"),
        "modifies_commented_line": result.get("modifies_commented_line"),
        "code_before": result.get("code_before"),
        "code_after": result.get("code_after"),
        "processing_error": result.get("processing_error"),
    }


def build_relevant_comment_record(
    row_id: int, row: pd.Series, result: dict[str, Any]
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "id": sanitize_for_csv(row.get("id")),
        "repo": sanitize_for_csv(row.get("repo")),
        "pr_number": sanitize_for_csv(row.get("pr_number")),
        "review_comment_id": sanitize_for_csv(row.get("review_comment_id")),
        "review_comment_html_url": sanitize_for_csv(
            row.get("review_comment_html_url")
        ),
        "commit_id": sanitize_for_csv(row.get("commit_id")),
        "commit_after": sanitize_for_csv(result.get("commit_after")),
        "path": sanitize_for_csv(row.get("path")),
        "line": sanitize_for_csv(row.get("line")),
        "original_line": sanitize_for_csv(row.get("original_line")),
        "body": sanitize_for_csv(row.get("body")),
        "code_before": sanitize_for_csv(result.get("code_before")),
        "code_after": sanitize_for_csv(result.get("code_after")),
    }


def is_relevant_comment(result: dict[str, Any]) -> bool:
    return (
        result.get("commit_after") != "-"
        and result.get("file_after") != "-"
        and result.get("modifies_commented_line") is True
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    comments = read_comments_input()
    print(f"[INFO] language: {LANGUAGE}")
    print(f"[INFO] target extension: {TARGET_FILE_EXTENSION}")
    print(f"[INFO] comments loaded: {len(comments)}")

    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    start_row_id = int(checkpoint.get("next_row_id", 0) or 0)

    stats = {
        "processed_rows_total": int(checkpoint.get("processed_rows_total", 0) or 0),
        "commit_after_found_total": int(
            checkpoint.get("commit_after_found_total", 0) or 0
        ),
        "commit_after_not_found_total": int(
            checkpoint.get("commit_after_not_found_total", 0) or 0
        ),
        "file_after_found_total": int(
            checkpoint.get("file_after_found_total", 0) or 0
        ),
        "modifies_commented_line_true_total": int(
            checkpoint.get("modifies_commented_line_true_total", 0) or 0
        ),
        "relevant_comments_total": int(
            checkpoint.get("relevant_comments_total", 0) or 0
        ),
    }

    if start_row_id > 0:
        print(f"[INFO] resuming from checkpoint: {CHECKPOINT_PATH}")
        print(f"[INFO] next row to process: {start_row_id}")
        print_stats("[INFO] checkpoint stats", stats)

    repo_to_pr_file = build_repo_to_pr_commits_file_map(PR_COMMITS_DIRS)
    print(f"[INFO] PR commit files found: {len(repo_to_pr_file)}")

    pr_file_cache: dict[str, dict[str, list[str]] | None] = {}
    pr_api_cache: dict[tuple[str, str], list[str] | None] = {}
    commit_api_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    processed_now = 0

    for row_id in tqdm(range(start_row_id, len(comments))):
        row = comments.iloc[row_id]

        try:
            result = process_comment_row(
                row=row,
                repo_to_pr_file=repo_to_pr_file,
                pr_file_cache=pr_file_cache,
                pr_api_cache=pr_api_cache,
                commit_api_cache=commit_api_cache,
            )
        except Exception as exc:
            result = empty_result(f"unexpected_error:{type(exc).__name__}:{exc}")

        record = build_output_record(row_id=row_id, row=row, result=result)
        append_jsonl_record(OUTPUT_JSONL_PATH, record)

        relevant = is_relevant_comment(result)
        if relevant:
            relevant_record = build_relevant_comment_record(
                row_id=row_id, row=row, result=result
            )
            append_csv_record(
                path=RELEVANT_COMMENTS_CSV_PATH,
                record=relevant_record,
                fieldnames=RELEVANT_CSV_FIELDS,
            )

        increment_stats(stats, result, is_relevant=relevant)
        processed_now += 1

        update_checkpoint(
            path=CHECKPOINT_PATH,
            next_row_id=row_id + 1,
            last_written_row_id=row_id,
            stats=stats,
        )

        if processed_now % CHECKPOINT_EVERY == 0:
            print(f"[CHECKPOINT] saved after {processed_now} newly processed rows")
            print_stats("[CHECKPOINT STATS]", stats)
            print(f"[CHECKPOINT] jsonl updated: {OUTPUT_JSONL_PATH}")
            print(f"[CHECKPOINT] relevant csv updated: {RELEVANT_COMMENTS_CSV_PATH}")
            print(f"[CHECKPOINT] checkpoint updated: {CHECKPOINT_PATH}")

    print("[DONE]")
    print(f"[INFO] jsonl output saved to: {OUTPUT_JSONL_PATH}")
    print(f"[INFO] relevant comments csv saved to: {RELEVANT_COMMENTS_CSV_PATH}")
    print(f"[INFO] checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"[INFO] rows processed in this run: {processed_now}")
    print_stats("[FINAL STATS]", stats)


if __name__ == "__main__":
    main()
