"""
Extract the function/method referenced by each relevant review comment before and
after the next pull-request commit.

This script is configured with constants at the top. It reads the outputs created
by the previous replication-package scripts:

1. mining_review_comments_only.py
   - pr_review_comments_<dataset>/*_pr_review_comments.csv
   - provides diff_hunk and file_content_at_comment_commit

2. mining_commit_after_unified.py
   - commit_after_<dataset>/filtered_comments_with_commit_after_<language>.jsonl
   - provides commit_after, diff_after, and the relevance flag

Change only LANGUAGE to switch between Java and Python runs.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from tree_sitter import Language, Parser
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "Missing dependency: tree-sitter. Install the parser dependencies before "
        "running this script, for example: pip install tree-sitter "
        "tree-sitter-java tree-sitter-python"
    ) from exc


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
        "tree_sitter_module": "tree_sitter_java",
        "declaration_node_types": {"method_declaration", "constructor_declaration"},
        "include_preceding_doc_block": True,
    },
    "python": {
        "repos_csv_stem": "500_random_python_repos",
        "target_file_extension": ".py",
        "tree_sitter_module": "tree_sitter_python",
        "declaration_node_types": {"function_definition"},
        "include_preceding_doc_block": False,
    },
}

if LANGUAGE not in LANGUAGE_CONFIG:
    valid_languages = ", ".join(sorted(LANGUAGE_CONFIG))
    raise ValueError(f"Unsupported LANGUAGE={LANGUAGE!r}. Use one of: {valid_languages}.")

REPOS_CSV_STEM = LANGUAGE_CONFIG[LANGUAGE]["repos_csv_stem"]
TARGET_FILE_EXTENSION = LANGUAGE_CONFIG[LANGUAGE]["target_file_extension"]

REVIEW_COMMENTS_DIR = f"pr_review_comments_{REPOS_CSV_STEM}"
REVIEW_COMMENTS_GLOB = "*_pr_review_comments.csv"

COMMIT_AFTER_DIR = f"commit_after_{REPOS_CSV_STEM}"
COMMIT_AFTER_JSONL_PATH = os.path.join(
    COMMIT_AFTER_DIR, f"filtered_comments_with_commit_after_{LANGUAGE}.jsonl"
)

OUTPUT_DIR = f"functions_before_after_{REPOS_CSV_STEM}"
OUTPUT_CSV_PATH = os.path.join(
    OUTPUT_DIR, f"relevant_comments_with_functions_{LANGUAGE}.csv"
)
FAILURES_CSV_PATH = os.path.join(
    OUTPUT_DIR, f"function_extraction_failures_{LANGUAGE}.csv"
)

INCLUDE_FILE_CONTENT_AFTER_IN_OUTPUT = True
INCLUDE_FILE_CONTENT_BEFORE_IN_OUTPUT = True
ALLOW_NAME_FALLBACK = True


# -----------------------------------------------------------------------------
# Tree-sitter setup
# -----------------------------------------------------------------------------

PARSER: Parser | None = None


def make_parser() -> Parser:
    parser = Parser()
    module_name = LANGUAGE_CONFIG[LANGUAGE]["tree_sitter_module"]

    try:
        parser_module = __import__(module_name)
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            f"Missing dependency: {module_name}. Install it before running this script."
        ) from exc

    language_capsule = parser_module.language()
    try:
        language = Language(language_capsule)
    except TypeError:
        language = language_capsule

    try:
        parser.language = language
    except AttributeError:
        parser.set_language(language)

    return parser


def get_parser() -> Parser:
    global PARSER
    if PARSER is None:
        PARSER = make_parser()
    return PARSER


# -----------------------------------------------------------------------------
# Generic parsing helpers
# -----------------------------------------------------------------------------


def walk_tree(node) -> Iterable[Any]:
    yield node
    for child in node.children:
        yield from walk_tree(child)


def node_text(source_bytes: bytes, node) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def declaration_node_types() -> set[str]:
    return set(LANGUAGE_CONFIG[LANGUAGE]["declaration_node_types"])


def parse_source(file_content: str):
    source_bytes = file_content.encode("utf-8")
    tree = get_parser().parse(source_bytes)
    return source_bytes, tree.root_node


# -----------------------------------------------------------------------------
# Diff helpers
# -----------------------------------------------------------------------------

HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


class DiffApplyError(Exception):
    """Raised when a unified diff cannot be applied to the expected content."""


def normalize_for_diff_compare(line: str) -> str:
    return line.rstrip("\r\n")


def get_last_changed_new_file_line_from_diff(diff: str) -> int | None:
    """Return the last changed line in the new-file side of a unified diff."""
    if not isinstance(diff, str) or not diff.strip():
        return None

    old_line = None
    new_line = None
    last_changed_line = None
    fallback_line_for_deletion = None
    deletion_pending = False

    for diff_line in diff.splitlines():
        hunk_match = HUNK_HEADER_RE.match(diff_line)
        if hunk_match:
            old_line = int(hunk_match.group("old_start"))
            new_line = int(hunk_match.group("new_start"))
            deletion_pending = False
            fallback_line_for_deletion = None
            continue

        if old_line is None or new_line is None:
            continue

        if diff_line.startswith("+++ ") or diff_line.startswith("--- "):
            continue

        if diff_line.startswith("+"):
            last_changed_line = new_line
            new_line += 1
            deletion_pending = False
        elif diff_line.startswith("-"):
            old_line += 1
            deletion_pending = True
            fallback_line_for_deletion = new_line
        else:
            if deletion_pending:
                fallback_line_for_deletion = new_line
                deletion_pending = False
            old_line += 1
            new_line += 1

    if last_changed_line is not None:
        return last_changed_line
    return fallback_line_for_deletion


def apply_unified_diff(original_content: str, diff: str) -> str:
    """Apply a unified diff to original_content and return the patched content."""
    if not isinstance(original_content, str):
        raise TypeError(
            f"original_content must be a string, got {type(original_content).__name__}"
        )
    if not isinstance(diff, str):
        raise TypeError(f"diff must be a string, got {type(diff).__name__}")
    if not diff.strip():
        return original_content

    original_lines = original_content.splitlines(keepends=True)
    diff_lines = diff.splitlines(keepends=True)
    result_lines = []
    original_index = 0
    diff_index = 0

    while diff_index < len(diff_lines):
        diff_line = diff_lines[diff_index]
        hunk_match = HUNK_HEADER_RE.match(diff_line)
        if not hunk_match:
            diff_index += 1
            continue

        old_start = int(hunk_match.group("old_start"))
        hunk_original_index = old_start - 1
        if hunk_original_index < original_index:
            raise DiffApplyError(
                f"Diff hunk is out of order or overlaps at original line {old_start}."
            )

        result_lines.extend(original_lines[original_index:hunk_original_index])
        original_index = hunk_original_index
        diff_index += 1

        while diff_index < len(diff_lines):
            diff_line = diff_lines[diff_index]
            if HUNK_HEADER_RE.match(diff_line):
                break
            if diff_line.startswith("\\") or not diff_line:
                diff_index += 1
                continue

            prefix = diff_line[0]
            line_content = diff_line[1:]

            if prefix == " ":
                if original_index >= len(original_lines):
                    raise DiffApplyError(
                        "Diff context line extends beyond the original file."
                    )
                original_line = original_lines[original_index]
                if normalize_for_diff_compare(original_line) != normalize_for_diff_compare(
                    line_content
                ):
                    raise DiffApplyError(
                        "Diff context line does not match the original file. "
                        f"Original line: {original_index + 1}"
                    )
                result_lines.append(original_line)
                original_index += 1
            elif prefix == "-":
                if original_index >= len(original_lines):
                    raise DiffApplyError(
                        "Diff removal extends beyond the original file."
                    )
                original_line = original_lines[original_index]
                if normalize_for_diff_compare(original_line) != normalize_for_diff_compare(
                    line_content
                ):
                    raise DiffApplyError(
                        "Diff removal line does not match the original file. "
                        f"Original line: {original_index + 1}"
                    )
                original_index += 1
            elif prefix == "+":
                result_lines.append(line_content)
            else:
                raise DiffApplyError(f"Unsupported diff prefix: {prefix!r}")

            diff_index += 1

    result_lines.extend(original_lines[original_index:])
    return "".join(result_lines)


# -----------------------------------------------------------------------------
# Function/method extraction helpers
# -----------------------------------------------------------------------------


def find_declaration_node_containing_line(file_content: str, line_number: int):
    """Find the smallest supported declaration node containing a 1-based line."""
    if not isinstance(file_content, str) or not file_content.strip():
        return None

    source_bytes, root = parse_source(file_content)
    candidates = []
    for node in walk_tree(root):
        if node.type not in declaration_node_types():
            continue
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        if start_line <= line_number <= end_line:
            candidates.append(node)

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item.end_byte - item.start_byte, item.start_byte))
    return candidates[0]


def extract_java_doc_start_byte_if_present(source_bytes: bytes, declaration_start_byte: int) -> int:
    before_declaration = source_bytes[:declaration_start_byte].decode(
        "utf-8", errors="ignore"
    )
    stripped_before = before_declaration.rstrip()

    doc_start = stripped_before.rfind("/**")
    doc_end = stripped_before.rfind("*/")
    if doc_start != -1 and doc_end != -1 and doc_end > doc_start:
        after_doc = stripped_before[doc_end + 2 :]
        if after_doc.strip() == "":
            return len(stripped_before[:doc_start].encode("utf-8"))

    return declaration_start_byte


def extract_python_decorator_start_byte_if_present(node) -> int:
    parent = getattr(node, "parent", None)
    if parent is not None and parent.type == "decorated_definition":
        return parent.start_byte
    return node.start_byte


def declaration_start_byte(source_bytes: bytes, node) -> int:
    if LANGUAGE == "java" and LANGUAGE_CONFIG[LANGUAGE]["include_preceding_doc_block"]:
        return extract_java_doc_start_byte_if_present(source_bytes, node.start_byte)
    if LANGUAGE == "python":
        return extract_python_decorator_start_byte_if_present(node)
    return node.start_byte


def extract_declaration_source_from_node(file_content: str, node) -> str:
    source_bytes = file_content.encode("utf-8")
    start_byte = declaration_start_byte(source_bytes, node)
    end_byte = node.end_byte
    if LANGUAGE == "python":
        parent = getattr(node, "parent", None)
        if parent is not None and parent.type == "decorated_definition":
            end_byte = parent.end_byte
    return source_bytes[start_byte:end_byte].decode("utf-8", errors="ignore")


def normalize_java_type(type_text: str) -> str:
    type_text = type_text.strip()
    type_text = re.sub(r"\bfinal\b", "", type_text)
    type_text = re.sub(r"\s+", " ", type_text).strip()
    type_text = re.sub(r"\s*<\s*", "<", type_text)
    type_text = re.sub(r"\s*>\s*", ">", type_text)
    type_text = re.sub(r"\s*,\s*", ",", type_text)
    type_text = re.sub(r"\s*\[\s*\]\s*", "[]", type_text)
    type_text = re.sub(r"\s*\.\.\.\s*", "...", type_text)
    return type_text


def extract_java_key(source_bytes: bytes, node) -> dict[str, Any] | None:
    name_node = node.child_by_field_name("name")
    params_node = node.child_by_field_name("parameters")
    type_node = node.child_by_field_name("type")

    if name_node is None:
        return None

    parameter_types = []
    if params_node is not None:
        for child in params_node.children:
            if child.type in {"formal_parameter", "spread_parameter"}:
                child_type_node = child.child_by_field_name("type")
                if child_type_node is not None:
                    parameter_types.append(
                        normalize_java_type(node_text(source_bytes, child_type_node))
                    )

    return {
        "language": "java",
        "node_type": node.type,
        "name": node_text(source_bytes, name_node),
        "return_type": normalize_java_type(node_text(source_bytes, type_node))
        if type_node is not None
        else "",
        "parameters": parameter_types,
    }


def extract_python_key(source_bytes: bytes, node) -> dict[str, Any] | None:
    name_node = node.child_by_field_name("name")
    params_node = node.child_by_field_name("parameters")
    return_type_node = node.child_by_field_name("return_type")

    if name_node is None:
        return None

    parameter_names = []
    if params_node is not None:
        params_text = node_text(source_bytes, params_node)
        parameter_names = extract_python_parameter_names(params_text)

    return {
        "language": "python",
        "node_type": node.type,
        "name": node_text(source_bytes, name_node),
        "return_type": node_text(source_bytes, return_type_node).strip()
        if return_type_node is not None
        else "",
        "parameters": parameter_names,
    }


def extract_python_parameter_names(parameters_text: str) -> list[str]:
    """Extract a stable parameter-name key from a Python parameter list."""
    try:
        parsed = ast.parse(f"def _tmp{parameters_text}:\n    pass\n")
    except SyntaxError:
        return [part.strip() for part in parameters_text.strip("()").split(",") if part.strip()]

    function = parsed.body[0]
    if not isinstance(function, ast.FunctionDef):
        return []

    args = function.args
    names = []
    names.extend(arg.arg for arg in args.posonlyargs)
    names.extend(arg.arg for arg in args.args)
    if args.vararg is not None:
        names.append("*" + args.vararg.arg)
    names.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg is not None:
        names.append("**" + args.kwarg.arg)
    return names


def extract_declaration_key(source_bytes: bytes, node) -> dict[str, Any] | None:
    if LANGUAGE == "java":
        return extract_java_key(source_bytes, node)
    if LANGUAGE == "python":
        return extract_python_key(source_bytes, node)
    raise ValueError(f"Unsupported language: {LANGUAGE}")


def extract_declaration_from_diff(diff: str, file_content: str) -> dict[str, Any] | None:
    if not isinstance(file_content, str) or not file_content.strip():
        return None

    last_changed_line = get_last_changed_new_file_line_from_diff(diff)
    if last_changed_line is None:
        return None

    node = find_declaration_node_containing_line(
        file_content=file_content,
        line_number=last_changed_line,
    )
    if node is None:
        return None

    source_bytes = file_content.encode("utf-8")
    key = extract_declaration_key(source_bytes, node)
    if key is None:
        return None

    return {
        "line_number": last_changed_line,
        "function_source": extract_declaration_source_from_node(file_content, node),
        "function_key": key,
    }


def extract_declaration_by_key(
    file_content: str,
    target_key: dict[str, Any],
    allow_name_fallback: bool = ALLOW_NAME_FALLBACK,
) -> str | None:
    if not isinstance(file_content, str) or not file_content.strip():
        return None
    if not isinstance(target_key, dict):
        return None

    source_bytes, root = parse_source(file_content)
    same_name_candidates = []

    for node in walk_tree(root):
        if node.type not in declaration_node_types():
            continue

        current_key = extract_declaration_key(source_bytes, node)
        if current_key is None:
            continue

        if current_key == target_key:
            return extract_declaration_source_from_node(file_content, node)

        if current_key.get("name") == target_key.get("name"):
            same_name_candidates.append(node)

    if allow_name_fallback and len(same_name_candidates) == 1:
        return extract_declaration_source_from_node(file_content, same_name_candidates[0])

    return None


def extract_before_and_after(
    file_content_before: str,
    diff_before: str,
    diff_after: str,
) -> dict[str, Any]:
    result = {
        "ok": False,
        "error_step": None,
        "error": None,
        "function_before": None,
        "function_after": None,
        "function_key": None,
        "file_content_after": None,
        "line_number_before": None,
    }

    before_info = extract_declaration_from_diff(diff_before, file_content_before)
    if before_info is None:
        result["error_step"] = "extract_function_before"
        result["error"] = "No containing function/method found for the review-comment diff hunk."
        return result

    result["function_before"] = before_info["function_source"]
    result["function_key"] = before_info["function_key"]
    result["line_number_before"] = before_info["line_number"]

    try:
        file_content_after = apply_unified_diff(file_content_before, diff_after)
    except Exception as exc:
        result["error_step"] = "apply_diff_after"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["file_content_after"] = file_content_after

    function_after = extract_declaration_by_key(
        file_content=file_content_after,
        target_key=before_info["function_key"],
    )
    if function_after is None:
        result["error_step"] = "extract_function_after"
        result["error"] = "The diff was applied, but the same function/method was not found afterward."
        return result

    result["function_after"] = function_after
    result["ok"] = True
    return result


# -----------------------------------------------------------------------------
# Input loading and joining
# -----------------------------------------------------------------------------


def read_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}") from exc
    return pd.DataFrame(records)


def read_review_comments() -> pd.DataFrame:
    input_dir = Path(REVIEW_COMMENTS_DIR)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Review-comments directory not found: {input_dir}")

    paths = sorted(input_dir.glob(REVIEW_COMMENTS_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {REVIEW_COMMENTS_GLOB!r} found in {input_dir}"
        )

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_review_comments_csv"] = str(path)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def normalize_join_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def add_join_key(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["repo", "pr_number", "review_comment_id", "commit_id", "path"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    result = frame.copy()
    result["join_key"] = result.apply(
        lambda row: "||".join(normalize_join_value(row.get(column)) for column in required),
        axis=1,
    )
    return result


def is_relevant_after_record(row: pd.Series) -> bool:
    return (
        row.get("commit_after") not in {None, "", "-"}
        and row.get("file_after") not in {None, "", "-"}
        and str(row.get("modifies_commented_line")).lower() == "true"
        and isinstance(row.get("diff_after"), str)
        and row.get("diff_after") != "-"
    )


def load_joined_input() -> pd.DataFrame:
    if not os.path.exists(COMMIT_AFTER_JSONL_PATH):
        raise FileNotFoundError(f"Commit-after JSONL not found: {COMMIT_AFTER_JSONL_PATH}")

    after_records = read_jsonl(COMMIT_AFTER_JSONL_PATH)
    if after_records.empty:
        raise ValueError(f"No records found in {COMMIT_AFTER_JSONL_PATH}")

    after_records = after_records[after_records.apply(is_relevant_after_record, axis=1)]
    after_records = add_join_key(after_records)

    review_comments = read_review_comments()
    review_comments = add_join_key(review_comments)

    keep_review_columns = [
        "join_key",
        "diff_hunk",
        "file_content_at_comment_commit",
        "file_line",
        "diff_line",
        "is_comment_valid",
        "source_review_comments_csv",
    ]
    missing_review_columns = [
        column for column in keep_review_columns if column not in review_comments.columns
    ]
    if missing_review_columns:
        raise ValueError(
            "The review-comment files do not contain the columns needed for function "
            f"extraction: {missing_review_columns}"
        )

    review_subset = review_comments[keep_review_columns].drop_duplicates("join_key")
    joined = after_records.merge(review_subset, on="join_key", how="left")
    joined = joined.reset_index(drop=True)
    return joined


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def sanitize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: str, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: sanitize_csv_value(record.get(key)) for key in fieldnames})


def output_fieldnames(include_file_content: bool = False) -> list[str]:
    fields = [
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
        "diff_hunk",
        "diff_after",
        "line_number_before",
        "function_key",
        "function_before",
        "function_after",
        "extraction_ok",
        "error_step",
        "error",
        "source_review_comments_csv",
    ]
    if INCLUDE_FILE_CONTENT_BEFORE_IN_OUTPUT:
        fields.append("file_content_at_comment_commit")
    if include_file_content:
        fields.append("file_content_after")
    return fields


def build_output_record(row: pd.Series, extraction: dict[str, Any]) -> dict[str, Any]:
    record = {
        "row_id": row.get("row_id"),
        "id": row.get("id"),
        "repo": row.get("repo"),
        "pr_number": row.get("pr_number"),
        "review_comment_id": row.get("review_comment_id"),
        "review_comment_html_url": row.get("review_comment_html_url"),
        "commit_id": row.get("commit_id"),
        "commit_after": row.get("commit_after"),
        "path": row.get("path"),
        "line": row.get("line"),
        "original_line": row.get("original_line"),
        "body": row.get("body"),
        "diff_hunk": row.get("diff_hunk"),
        "diff_after": row.get("diff_after"),
        "line_number_before": extraction.get("line_number_before"),
        "function_key": extraction.get("function_key"),
        "function_before": extraction.get("function_before"),
        "function_after": extraction.get("function_after"),
        "extraction_ok": extraction.get("ok"),
        "error_step": extraction.get("error_step"),
        "error": extraction.get("error"),
        "source_review_comments_csv": row.get("source_review_comments_csv"),
    }
    if INCLUDE_FILE_CONTENT_BEFORE_IN_OUTPUT:
        record["file_content_at_comment_commit"] = row.get("file_content_at_comment_commit")
    if INCLUDE_FILE_CONTENT_AFTER_IN_OUTPUT:
        record["file_content_after"] = extraction.get("file_content_after")
    return record


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_joined_input()
    print(f"[INFO] language: {LANGUAGE}")
    print(f"[INFO] joined relevant comments: {len(data)}")

    output_records = []
    failure_records = []

    missing_join_data = 0
    success_count = 0

    for index, row in data.iterrows():
        file_content_before = row.get("file_content_at_comment_commit")
        diff_before = row.get("diff_hunk")
        diff_after = row.get("diff_after")
        path = row.get("path")

        if not isinstance(path, str) or not path.endswith(TARGET_FILE_EXTENSION):
            extraction = {
                "ok": False,
                "error_step": "input_validation",
                "error": "Record does not target the configured file extension.",
                "function_before": None,
                "function_after": None,
                "function_key": None,
                "file_content_after": None,
                "line_number_before": None,
            }
        elif not isinstance(file_content_before, str) or not isinstance(diff_before, str):
            missing_join_data += 1
            extraction = {
                "ok": False,
                "error_step": "join_review_comment_data",
                "error": "Missing diff_hunk or file_content_at_comment_commit after joining inputs.",
                "function_before": None,
                "function_after": None,
                "function_key": None,
                "file_content_after": None,
                "line_number_before": None,
            }
        else:
            extraction = extract_before_and_after(
                file_content_before=file_content_before,
                diff_before=diff_before,
                diff_after=diff_after,
            )

        record = build_output_record(row, extraction)
        if extraction.get("ok"):
            success_count += 1
            output_records.append(record)
        else:
            failure_records.append(record)

        if (index + 1) % 100 == 0:
            print(
                f"[PROGRESS] processed={index + 1} | "
                f"success={success_count} | failures={len(failure_records)}"
            )

    fields = output_fieldnames(include_file_content=INCLUDE_FILE_CONTENT_AFTER_IN_OUTPUT)
    write_csv(OUTPUT_CSV_PATH, output_records, fields)
    write_csv(FAILURES_CSV_PATH, failure_records, fields)

    print("[DONE]")
    print(f"[INFO] successful extractions: {success_count}")
    print(f"[INFO] failed extractions: {len(failure_records)}")
    print(f"[INFO] records with missing joined review-comment data: {missing_join_data}")
    print(f"[INFO] output saved to: {OUTPUT_CSV_PATH}")
    print(f"[INFO] failures saved to: {FAILURES_CSV_PATH}")


if __name__ == "__main__":
    main()
