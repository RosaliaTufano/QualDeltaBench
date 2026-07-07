#!/usr/bin/env python3

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

import fcntl

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# The script expects this directory to contain:
# - data/java_dataset.json
# - data/python_dataset.json
# - repos/                        # Java instance repositories
# - repos_python/                 # Python instance repositories
# - prompts/prompt_before_only_agentic.txt
# - prompts/prompt_before_only_agentic_without_quality_aspects.txt
# - quality_aspects.txt
# - optional .env file with OPENAI_API_KEY and/or OPENROUTER_API_KEY
# - optional key files: openai_apikey.txt, openrouter_token.txt, openrouter_apikey.txt
RUNNING_EXPERIMENTS_DIR = SCRIPT_DIR / "running_experimens"

# Options: "java", "python"
LANGUAGE = "python"

# Options:
# - "with_qa": the model receives the target quality aspect.
# - "without_qa": the model receives all quality aspects and must select the relevant one(s).
EXPERIMENT_TYPE = "without_qa"

# Options: "openai", "openrouter"
# Use "openai" for OpenAI-hosted GPT models and "openrouter" for OpenRouter models
# such as MiniMax or Gemini.
MODEL_PROVIDER = "openrouter"

# Examples:
# - OpenAI:     "openai/gpt-5.4-mini"
# - OpenRouter: "minimax/minimax-m3"
# - OpenRouter: "google/gemini-2.5-pro"
MODEL = "minimax/minimax-m3"

# Optional display alias used in output folder names. Leave empty to infer it from MODEL.
MODEL_OUTPUT_ALIAS = ""

# None = run all instances. Use an integer to run only the first N instances.
NUM_INSTANCES_TO_RUN: int | None = None

# Number of complete repetitions of the experiment.
NUM_RUNS = 10

MAX_STEPS = 10
COST_LIMIT = 0.10

VERBOSE_AGENT_OUTPUT = False
STOP_ON_FAILURE = False

# If True, remove run_i/results.jsonl at the beginning of each run.
# This avoids duplicated rows when relaunching the script.
OVERWRITE_RESULTS_JSONL = True

# If True, reuse existing successful results and retry only selected statuses.
RESUME_RESULTS = False
RETRY_STATUSES = {"failed", "parse_failed"}


# ============================================================
# Derived paths and validation
# ============================================================

load_dotenv(RUNNING_EXPERIMENTS_DIR / ".env")

LANGUAGE_CONFIGS = {
    "python": {
        "dataset_file": RUNNING_EXPERIMENTS_DIR / "data" / "python_dataset.json",
        "repos_dir": RUNNING_EXPERIMENTS_DIR / "repos_python",
        "experiment_prefix": "before_only_python_",
    },
    "java": {
        "dataset_file": RUNNING_EXPERIMENTS_DIR / "data" / "java_dataset.json",
        "repos_dir": RUNNING_EXPERIMENTS_DIR / "repos",
        "experiment_prefix": "before_only_",
    },
}

EXPERIMENT_CONFIGS = {
    "with_qa": {
        "experiment_name": "runs_with_qa",
        "prompt_template_file": RUNNING_EXPERIMENTS_DIR / "prompts" / "prompt_before_only_agentic.txt",
    },
    "without_qa": {
        "experiment_name": "runs_without_qa",
        "prompt_template_file": RUNNING_EXPERIMENTS_DIR / "prompts" / "prompt_before_only_agentic_without_quality_aspects.txt",
    },
}

if LANGUAGE not in LANGUAGE_CONFIGS:
    raise ValueError(
        f"Unsupported LANGUAGE={LANGUAGE!r}. Supported values: {list(LANGUAGE_CONFIGS)}"
    )

if EXPERIMENT_TYPE not in EXPERIMENT_CONFIGS:
    raise ValueError(
        f"Unsupported EXPERIMENT_TYPE={EXPERIMENT_TYPE!r}. "
        f"Supported values: {list(EXPERIMENT_CONFIGS)}"
    )

if MODEL_PROVIDER not in {"openai", "openrouter"}:
    raise ValueError("MODEL_PROVIDER must be either 'openai' or 'openrouter'.")

DATASET_FILE = LANGUAGE_CONFIGS[LANGUAGE]["dataset_file"]
REPOS_DIR = LANGUAGE_CONFIGS[LANGUAGE]["repos_dir"]
QUALITY_ASPECTS_FILE = RUNNING_EXPERIMENTS_DIR / "quality_aspects.txt"
PROMPT_TEMPLATE_FILE = EXPERIMENT_CONFIGS[EXPERIMENT_TYPE]["prompt_template_file"]


def slugify_for_path(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "model"


def output_model_name(model: str) -> str:
    if MODEL_OUTPUT_ALIAS.strip():
        return slugify_for_path(MODEL_OUTPUT_ALIAS.strip())

    lowered = model.lower()
    if "minimax" in lowered:
        return "minimax"
    if "gemini" in lowered:
        return "gemini"
    if "gpt" in lowered or "openai" in lowered:
        return "gpt"

    return slugify_for_path(model)


OUTPUT_MODEL_NAME = output_model_name(MODEL)
BASE_EXPERIMENT_NAME = (
    f"{LANGUAGE_CONFIGS[LANGUAGE]['experiment_prefix']}"
    f"{EXPERIMENT_CONFIGS[EXPERIMENT_TYPE]['experiment_name']}"
)
EXPERIMENT_NAME = f"{BASE_EXPERIMENT_NAME}_{OUTPUT_MODEL_NAME}"
EXPERIMENT_ROOT_DIR = RUNNING_EXPERIMENTS_DIR / EXPERIMENT_NAME

# The original dataset layout contains:
# repos/<instance_id>/before/
# repos/<instance_id>/after/
#
# This experiment only exposes before/ to the agent, temporarily renamed as:
# repos/<instance_id>/repo_snapshot/
ORIGINAL_SNAPSHOT_DIRNAME = "before"
ORIGINAL_AFTER_DIRNAME = "after"
INSPECTION_SNAPSHOT_DIRNAME = "repo_snapshot"

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENAI_KEY_FILES = [RUNNING_EXPERIMENTS_DIR / "openai_apikey.txt"]
OPENROUTER_KEY_FILES = [
    RUNNING_EXPERIMENTS_DIR / "openrouter_token.txt",
    RUNNING_EXPERIMENTS_DIR / "openrouter_apikey.txt",
]

# Separate mini-swe-agent global configuration directory from the experiment folder.
MINI_GLOBAL_CONFIG_DIR = RUNNING_EXPERIMENTS_DIR / ".mini-swe-agent-config"


def default_lock_dir_for_repos(repos_dir: Path) -> Path:
    resolved = repos_dir.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved.name).strip("_") or "repos"
    return RUNNING_EXPERIMENTS_DIR / ".locks" / f"{safe_name}_{digest}"


LOCK_DIR = default_lock_dir_for_repos(REPOS_DIR)

# In this before-only experiment, every instance is expected to contain an issue.
CORRECT_ANSWER = "Yes"


# ============================================================
# Dataset helpers
# ============================================================

def load_json(dataset_file: Path) -> list[dict]:
    if not dataset_file.exists():
        raise FileNotFoundError(f"Missing dataset file: {dataset_file}")

    try:
        data = json.loads(dataset_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file {dataset_file}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Expected {dataset_file} to contain a JSON list of objects, "
            f"but got {type(data).__name__}"
        )

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(
                f"Expected item {i} in {dataset_file} to be an object, "
                f"but got {type(row).__name__}"
            )

    return data


# ============================================================
# Quality aspect helpers
# ============================================================

def load_quality_aspect_descriptions(quality_aspects_file: Path) -> dict[str, str]:
    if not quality_aspects_file.exists():
        raise FileNotFoundError(f"Missing quality aspects file: {quality_aspects_file}")

    descriptions = {}

    for line_number, line in enumerate(
        quality_aspects_file.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        if " -- " not in line:
            raise ValueError(
                f"Invalid format in {quality_aspects_file} at line {line_number}: {line!r}. "
                "Expected format: '<quality aspect> -- <description>'"
            )

        quality_aspect, description = line.split(" -- ", 1)

        quality_aspect = quality_aspect.strip()
        description = description.strip()

        prefix = "Characterized by concepts like:"
        if description.startswith(prefix):
            description = description[len(prefix):].strip()

        if not quality_aspect:
            raise ValueError(
                f"Empty quality aspect name in {quality_aspects_file} at line {line_number}"
            )

        if not description:
            raise ValueError(
                f"Empty description for {quality_aspect!r} in {quality_aspects_file} "
                f"at line {line_number}"
            )

        descriptions[quality_aspect] = description

    return descriptions


def format_quality_aspects_for_prompt(
    quality_aspect_descriptions: dict[str, str],
) -> str:
    """
    Formats all quality aspects for prompt templates that contain a
    {quality_aspects} placeholder.

    If the prompt template hardcodes the quality aspects, this value is harmless:
    build_task_from_template still provides it, but the template simply will not use it.
    """

    return "\n".join(
        f"- {quality_aspect}: {description}"
        for quality_aspect, description in quality_aspect_descriptions.items()
    )


def read_api_key_from_files_or_env(*, env_name: str, key_files: list[Path]) -> str:
    for key_file in key_files:
        if key_file.exists():
            candidate = "".join(
                line.strip() for line in key_file.read_text(encoding="utf-8").splitlines()
            ).strip()
            if candidate:
                return candidate

    candidate = os.environ.get(env_name, "").strip()
    if candidate:
        return candidate

    raise RuntimeError(
        f"Missing API key. Set {env_name} or create one of: "
        f"{', '.join(str(path) for path in key_files)}"
    )


def get_model_api_key() -> tuple[str, str]:
    if MODEL_PROVIDER == "openai":
        return OPENAI_API_KEY_ENV, read_api_key_from_files_or_env(
            env_name=OPENAI_API_KEY_ENV,
            key_files=OPENAI_KEY_FILES,
        )

    if MODEL_PROVIDER == "openrouter":
        return OPENROUTER_API_KEY_ENV, read_api_key_from_files_or_env(
            env_name=OPENROUTER_API_KEY_ENV,
            key_files=OPENROUTER_KEY_FILES,
        )

    raise ValueError(f"Unsupported MODEL_PROVIDER={MODEL_PROVIDER!r}")


# ============================================================
# Prompt helpers
# ============================================================

def load_prompt_template(prompt_template_file: Path) -> str:
    if not prompt_template_file.exists():
        raise FileNotFoundError(f"Missing prompt template file: {prompt_template_file}")

    return prompt_template_file.read_text(encoding="utf-8")


def build_task_from_template(
    template: str,
    *,
    instance_id: str,
    snapshot_dirname: str,
    file_path: str,
    function_name: str,
    quality_aspect: str = "",
    quality_aspect_description: str = "",
    quality_aspects: str = "",
    repo: str | None = None,
    pr_number: int | None = None,
    commit_before: str | None = None,
    commit_after: str | None = None,
) -> str:
    return template.format(
        instance_id=instance_id,
        snapshot_dirname=snapshot_dirname,
        file_path=file_path,
        function_name=function_name,
        quality_aspect=quality_aspect,
        quality_aspect_description=quality_aspect_description,
        quality_aspects=quality_aspects,
        repo=repo or "",
        pr_number=pr_number or "",
        commit_before=commit_before or "",
        commit_after=commit_after or "",
    )


# ============================================================
# Agent response helpers
# ============================================================

def parse_agent_response(response: str, experiment_type: str) -> dict:
    if experiment_type == "with_qa":
        return parse_with_qa_agent_response(response)

    if experiment_type == "without_qa":
        return parse_without_qa_agent_response(response)

    raise ValueError(f"Unsupported experiment_type={experiment_type!r}")


def parse_issue_value(value: str | None) -> str:
    if value is None:
        raise ValueError("Could not parse Issue field from response.")

    normalized = value.strip().lower()

    if normalized == "yes":
        return "Yes"

    if normalized == "no":
        return "No"

    raise ValueError(f"Could not parse a valid Issue value from response: {value!r}")


def normalize_list_item(line: str) -> str:
    line = line.strip()

    if line.startswith("-"):
        line = line[1:].strip()

    return line


def is_none_marker(value: str) -> bool:
    return value.strip().lower() in {"none", "n/a", "na", "null"}


def parse_with_qa_agent_response(response: str) -> dict:
    """
    Parses a with_qa response shaped like:

    Issue: Yes

    Evidence: specific code-level evidence

    or:

    Issue: No

    Evidence: None
    """

    issue = None
    evidence = []

    current_section = None

    for raw_line in response.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        lower_line = line.lower()

        if lower_line.startswith("issue:"):
            issue = line.split(":", 1)[1].strip()
            current_section = None
            continue

        if lower_line.startswith("evidence:"):
            first_evidence = line.split(":", 1)[1].strip()
            current_section = "evidence"

            if first_evidence and not is_none_marker(first_evidence):
                evidence.append(first_evidence)

            continue

        if current_section == "evidence":
            evidence_item = normalize_list_item(line)

            if evidence_item and not is_none_marker(evidence_item):
                evidence.append(evidence_item)

            continue

    parsed_issue = parse_issue_value(issue)

    if parsed_issue == "No":
        evidence = []

    return {
        "issue": parsed_issue,
        "selected_quality_aspects": [],
        "evidence": evidence,
    }


def parse_without_qa_agent_response(response: str) -> dict:
    """
    Parses a without_qa response shaped like:

    Issue: Yes

    Quality aspects:
    - Correctness
    - Maintainability & Code Structure

    Evidence:
    - specific code-level evidence

    or:

    Issue: No

    Quality aspects:
    - None

    Evidence:
    - None
    """

    issue = None
    selected_quality_aspects = []
    evidence = []

    current_section = None

    for raw_line in response.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        lower_line = line.lower()

        if lower_line.startswith("issue:"):
            issue = line.split(":", 1)[1].strip()
            current_section = None
            continue

        if lower_line.startswith("quality aspects:"):
            current_section = "quality_aspects"
            continue

        if lower_line.startswith("evidence:"):
            first_evidence = line.split(":", 1)[1].strip()
            current_section = "evidence"

            if first_evidence and not is_none_marker(first_evidence):
                evidence.append(first_evidence)

            continue

        if current_section == "quality_aspects":
            quality_aspect = normalize_list_item(line)

            if quality_aspect and not is_none_marker(quality_aspect):
                selected_quality_aspects.append(quality_aspect)

            continue

        if current_section == "evidence":
            evidence_item = normalize_list_item(line)

            if evidence_item and not is_none_marker(evidence_item):
                evidence.append(evidence_item)

            continue

    parsed_issue = parse_issue_value(issue)

    if parsed_issue == "Yes" and not selected_quality_aspects:
        raise ValueError("Issue is Yes but no selected quality aspects were parsed.")

    if parsed_issue == "No":
        selected_quality_aspects = []
        evidence = []

    return {
        "issue": parsed_issue,
        "selected_quality_aspects": selected_quality_aspects,
        "evidence": evidence,
    }


# ============================================================
# Trajectory/output helpers
# ============================================================

def extract_final_output_from_trajectory(trajectory_file: Path) -> str | None:
    """
    Tries to extract the final submitted output from mini-swe-agent trajectory.

    The exact trajectory schema can vary across versions, so this function
    searches recursively for messages with role == "exit".
    """

    if not trajectory_file.exists():
        return None

    try:
        data = json.loads(trajectory_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    def walk(obj):
        if isinstance(obj, dict):
            if obj.get("role") == "exit" and isinstance(obj.get("content"), str):
                yield obj["content"]

            for value in obj.values():
                yield from walk(value)

        elif isinstance(obj, list):
            for item in obj:
                yield from walk(item)

    exits = list(walk(data))

    if not exits:
        return None

    return exits[-1].strip()


def strip_completion_marker(output: str) -> str:
    """
    Removes the mini-swe-agent completion marker if it is present in the final output.
    """

    marker = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    lines = output.splitlines()

    if lines and lines[0].strip() == marker:
        return "\n".join(lines[1:]).strip()

    return output.strip()


def save_final_output_if_available(
    *,
    trajectory_file: Path,
    final_output_file: Path,
) -> None:
    final_output = extract_final_output_from_trajectory(trajectory_file)

    if final_output is None:
        print(f"No final output found in trajectory: {trajectory_file}")
        return

    final_output = strip_completion_marker(final_output)
    final_output_file.write_text(final_output + "\n", encoding="utf-8")


# ============================================================
# Usage helpers
# ============================================================

def extract_step_and_cost_from_log(log_file: Path) -> dict:
    """
    Extracts the last observed mini-swe-agent step and cost from the live log.

    Expected line shape:
    mini-swe-agent (step 3, $0.01):
    """

    if not log_file.exists():
        return {
            "agent_steps": None,
            "agent_cost": None,
        }

    pattern = re.compile(
        r"mini-swe-agent\s+\(step\s+(\d+),\s+\$([0-9.]+)\):"
    )

    last_step = None
    last_cost = None

    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            last_step = int(match.group(1))
            last_cost = float(match.group(2))

    return {
        "agent_steps": last_step,
        "agent_cost": last_cost,
    }


def format_usage_for_print(usage: dict) -> str:
    steps = usage.get("agent_steps")
    cost = usage.get("agent_cost")

    steps_text = str(steps) if steps is not None else "unknown"
    cost_text = f"${cost:.4f}" if cost is not None else "unknown"

    return f"Steps: {steps_text}, Cost: {cost_text}"


# ============================================================
# JSONL result helpers
# ============================================================

def append_jsonl_record(output_file: Path, record: dict) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_records_by_id(output_file: Path) -> dict[int, dict]:
    records = {}

    if not output_file.exists():
        return records

    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            records[record["id"]] = record

    return records


def write_jsonl_records_in_dataset_order(
    output_file: Path,
    records_by_id: dict[int, dict],
    selected_instances: list[dict],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")

    with temp_file.open("w", encoding="utf-8") as f:
        for instance in selected_instances:
            row_id = instance["row_id"]
            if row_id in records_by_id:
                f.write(json.dumps(records_by_id[row_id], ensure_ascii=False) + "\n")

    temp_file.replace(output_file)


def should_skip_existing_record(record: dict | None) -> bool:
    if record is None:
        return False

    return record.get("status") not in RETRY_STATUSES


def save_aggregated_result_record(
    *,
    instance: dict,
    final_output_file: Path,
    live_log_file: Path,
    aggregated_results_file: Path,
    run_number: int,
    status: str,
    error_message: str | None = None,
) -> dict:
    """
    Appends one JSONL record for the instance.

    In this before-only experiment:
    - correct_answer is always "Yes"
    - is_answer_correct is True if and only if the agent outputs Issue: Yes
    """

    response = ""
    issue = None
    selected_quality_aspects = []
    evidence = []
    parse_error = None

    if final_output_file.exists():
        response = final_output_file.read_text(encoding="utf-8").strip()

    if status == "success":
        try:
            parsed_response = parse_agent_response(response, EXPERIMENT_TYPE)
            issue = parsed_response["issue"]
            selected_quality_aspects = parsed_response["selected_quality_aspects"]
            evidence = parsed_response["evidence"]
        except Exception as exc:
            parse_error = str(exc)
            status = "parse_failed"
            issue = None
            selected_quality_aspects = []
            evidence = []

    correct_answer = CORRECT_ANSWER
    is_answer_correct = issue == correct_answer

    usage = extract_step_and_cost_from_log(live_log_file)

    record = {
        "run_number": run_number,
        "id": instance["row_id"],
        "experiment_type": EXPERIMENT_TYPE,
        "classification": instance.get("classification"),
        "model_response_raw": response,
        "issue": issue,
        "selected_quality_aspects": selected_quality_aspects,
        "evidence": evidence,
        "correct_answer": correct_answer,
        "is_answer_correct": is_answer_correct,
        "status": status,
        "error_message": error_message,
        "parse_error": parse_error,
        "agent_steps": usage["agent_steps"],
        "agent_cost": usage["agent_cost"],
    }

    append_jsonl_record(aggregated_results_file, record)
    return record


# ============================================================
# Validation helpers
# ============================================================

def validate_instance_files(
    *,
    root_dir: Path,
    snapshot_dirname: str,
    file_path: str,
) -> None:
    snapshot_file = root_dir / snapshot_dirname / file_path

    if not snapshot_file.exists():
        raise FileNotFoundError(f"Missing snapshot file: {snapshot_file}")


# ============================================================
# Snapshot directory helpers
# ============================================================

def rename_snapshot_dir_for_inspection(root_dir: Path) -> bool:
    """
    Temporarily renames before/ to repo_snapshot/.

    Returns True if a rename was performed.

    If repo_snapshot/ already exists, the function assumes the repository is already
    in the desired inspection layout and does not rename anything.
    """

    original_dir = root_dir / ORIGINAL_SNAPSHOT_DIRNAME
    inspection_dir = root_dir / INSPECTION_SNAPSHOT_DIRNAME

    if inspection_dir.exists():
        return False

    if not original_dir.exists():
        raise FileNotFoundError(f"Missing directory: {original_dir}")

    original_dir.rename(inspection_dir)
    return True


def restore_snapshot_dir_after_inspection(root_dir: Path) -> None:
    """
    Restores repo_snapshot/ back to before/.
    """

    original_dir = root_dir / ORIGINAL_SNAPSHOT_DIRNAME
    inspection_dir = root_dir / INSPECTION_SNAPSHOT_DIRNAME

    if not inspection_dir.exists():
        return

    if original_dir.exists():
        raise FileExistsError(
            f"Cannot restore snapshot directory because target already exists: {original_dir}"
        )

    inspection_dir.rename(original_dir)


def hide_after_snapshot_for_inspection(root_dir: Path, run_dir: Path, instance_id: str) -> Path | None:
    """
    Moves after/ outside the repository root while the before-only agent runs.

    The agent cwd is root_dir, so this keeps the visible repository layout aligned
    with the prompt: only repo_snapshot/ should be available for inspection.
    """

    after_dir = root_dir / ORIGINAL_AFTER_DIRNAME
    hidden_after_dir = run_dir / f"{instance_id}.hidden_after_snapshot"

    if not after_dir.exists():
        return None

    if hidden_after_dir.exists():
        raise FileExistsError(
            f"Cannot hide after snapshot because temporary directory already exists: "
            f"{hidden_after_dir}"
        )

    after_dir.rename(hidden_after_dir)
    return hidden_after_dir


def restore_after_snapshot_after_inspection(root_dir: Path, hidden_after_dir: Path | None) -> None:
    """
    Restores the hidden after/ snapshot after the before-only agent finishes.
    """

    if hidden_after_dir is None or not hidden_after_dir.exists():
        return

    after_dir = root_dir / ORIGINAL_AFTER_DIRNAME

    if after_dir.exists():
        raise FileExistsError(
            f"Cannot restore after snapshot because target already exists: {after_dir}"
        )

    hidden_after_dir.rename(after_dir)


@contextmanager
def instance_repo_lock(instance_id: str):
    lock_dir = LOCK_DIR
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / f"{instance_id}.lock"

    with lock_file.open("w", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ============================================================
# Runner
# ============================================================

def run_instance(
    *,
    instance: dict,
    quality_aspect_descriptions: dict[str, str],
    run_number: int,
    run_root_dir: Path,
    aggregated_results_file: Path,
) -> dict | None:
    instance_id = str(instance["row_id"])
    root_dir = REPOS_DIR / instance_id

    run_dir = run_root_dir / instance_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trajectory_file = run_dir / f"{instance_id}.traj.json"
    live_log_file = run_dir / f"{instance_id}.live.log"
    final_output_file = run_dir / f"{instance_id}.final_output.txt"
    task_prompt_file = run_dir / f"{instance_id}.prompt.txt"

    snapshot_was_renamed = False
    hidden_after_dir = None
    repo_lock = None

    try:
        file_path = instance["path"]
        function_name = instance["signature"]

        quality_aspect = ""
        quality_aspect_description = ""

        if EXPERIMENT_TYPE == "with_qa":
            quality_aspect = instance.get("classification")

            if not quality_aspect:
                raise ValueError(f"Missing classification for instance row_id={instance_id}")

            if quality_aspect not in quality_aspect_descriptions:
                available = ", ".join(sorted(quality_aspect_descriptions))
                raise ValueError(
                    f"Quality aspect {quality_aspect!r} for instance row_id={instance_id} "
                    f"was not found in {QUALITY_ASPECTS_FILE}. "
                    f"Available quality aspects: {available}"
                )

            quality_aspect_description = quality_aspect_descriptions[quality_aspect]

        quality_aspects = format_quality_aspects_for_prompt(quality_aspect_descriptions)

        repo = instance.get("repo")
        pr_number = instance.get("pr_number")
        commit_before = instance.get("commit_id")
        commit_after = instance.get("commit_after")

        repo_lock = instance_repo_lock(instance_id)
        repo_lock.__enter__()

        validate_instance_files(
            root_dir=root_dir,
            snapshot_dirname=ORIGINAL_SNAPSHOT_DIRNAME,
            file_path=file_path,
        )

        snapshot_was_renamed = rename_snapshot_dir_for_inspection(root_dir)
        hidden_after_dir = hide_after_snapshot_for_inspection(root_dir, run_dir, instance_id)

        prompt_template = load_prompt_template(PROMPT_TEMPLATE_FILE)

        task = build_task_from_template(
            prompt_template,
            instance_id=instance_id,
            snapshot_dirname=INSPECTION_SNAPSHOT_DIRNAME,
            file_path=file_path,
            function_name=function_name,
            quality_aspect=quality_aspect,
            quality_aspect_description=quality_aspect_description,
            quality_aspects=quality_aspects,
            repo=repo,
            pr_number=pr_number,
            commit_before=commit_before,
            commit_after=commit_after,
        )

        task_prompt_file.write_text(task, encoding="utf-8")

        cmd = [
            "mini",
            "--model",
            MODEL,
            "--task",
            task,
            "--yolo",
            "--exit-immediately",
            "--cost-limit",
            str(COST_LIMIT),
            "-o",
            str(trajectory_file),
            "-c",
            "mini.yaml",
            "-c",
            f"agent.step_limit={MAX_STEPS}",
        ]

        if MODEL_PROVIDER == "openrouter":
            cmd.extend(["--model-class", "openrouter"])

        api_key_env_name, api_key = get_model_api_key()
        child_env = os.environ.copy()
        child_env["MSWEA_GLOBAL_CONFIG_DIR"] = str(MINI_GLOBAL_CONFIG_DIR)
        child_env[api_key_env_name] = api_key
        child_env["MSWEA_COST_TRACKING"] = "ignore_errors"
        child_env["MSWEA_MODEL_NAME"] = MODEL
        child_env["MSWEA_CONFIGURED"] = "1"

        print(f"[run_{run_number}][{instance_id}] Inspecting repository snapshot...")
        print(f"[run_{run_number}][{instance_id}] Snapshot: {INSPECTION_SNAPSHOT_DIRNAME}")
        print(f"[run_{run_number}][{instance_id}] File: {file_path}")
        print(f"[run_{run_number}][{instance_id}] Function: {function_name}")

        if EXPERIMENT_TYPE == "with_qa":
            print(f"[run_{run_number}][{instance_id}] Quality aspect: {quality_aspect}")
        else:
            print(f"[run_{run_number}][{instance_id}] Quality aspects: all")

        with live_log_file.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=root_dir,
                text=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                env=child_env,
            )

            assert process.stdout is not None

            for line in process.stdout:
                if VERBOSE_AGENT_OUTPUT:
                    print(line, end="")

                log_file.write(line)
                log_file.flush()

            return_code = process.wait()

        save_final_output_if_available(
            trajectory_file=trajectory_file,
            final_output_file=final_output_file,
        )

        if return_code != 0:
            usage = extract_step_and_cost_from_log(live_log_file)

            print(
                f"[run_{run_number}][{instance_id}] Failed with exit code {return_code}. "
                f"{format_usage_for_print(usage)}"
            )

            record = save_aggregated_result_record(
                instance=instance,
                final_output_file=final_output_file,
                live_log_file=live_log_file,
                aggregated_results_file=aggregated_results_file,
                run_number=run_number,
                status="failed",
                error_message=f"mini-swe-agent failed with exit code {return_code}",
            )

            if STOP_ON_FAILURE:
                raise RuntimeError(
                    f"Instance {instance_id} failed with exit code {return_code}"
                )

            return record

        record = save_aggregated_result_record(
            instance=instance,
            final_output_file=final_output_file,
            live_log_file=live_log_file,
            aggregated_results_file=aggregated_results_file,
            run_number=run_number,
            status="success",
        )

        usage = extract_step_and_cost_from_log(live_log_file)

        print(
            f"[run_{run_number}][{instance_id}] Done. "
            f"{format_usage_for_print(usage)}"
        )

        return record

    except Exception as exc:
        error_message = str(exc)

        print(f"[run_{run_number}][{instance_id}] Unexpected failure: {error_message}")

        save_final_output_if_available(
            trajectory_file=trajectory_file,
            final_output_file=final_output_file,
        )

        try:
            record = save_aggregated_result_record(
                instance=instance,
                final_output_file=final_output_file,
                live_log_file=live_log_file,
                aggregated_results_file=aggregated_results_file,
                run_number=run_number,
                status="failed",
                error_message=error_message,
            )
            return record
        except Exception as record_exc:
            print(
                f"[run_{run_number}][{instance_id}] Could not save failure record: "
                f"{record_exc}"
            )

        if STOP_ON_FAILURE:
            raise

        return None

    finally:
        try:
            try:
                restore_after_snapshot_after_inspection(root_dir, hidden_after_dir)
            finally:
                if snapshot_was_renamed:
                    restore_snapshot_dir_after_inspection(root_dir)
        finally:
            if repo_lock is not None:
                repo_lock.__exit__(None, None, None)


def run_experiment_once(
    *,
    run_number: int,
    selected_instances: list[dict],
    quality_aspect_descriptions: dict[str, str],
) -> None:
    run_root_dir = EXPERIMENT_ROOT_DIR / f"run_{run_number}"
    run_root_dir.mkdir(parents=True, exist_ok=True)

    aggregated_results_file = run_root_dir / "results.jsonl"

    if OVERWRITE_RESULTS_JSONL and not RESUME_RESULTS and aggregated_results_file.exists():
        aggregated_results_file.unlink()

    existing_records = load_jsonl_records_by_id(aggregated_results_file) if RESUME_RESULTS else {}

    print("=" * 80)
    print(f"Starting run_{run_number}")
    print(f"Run directory: {run_root_dir}")
    print(f"Aggregated results file: {aggregated_results_file}")
    if RESUME_RESULTS:
        print(f"Resume mode: retry statuses {sorted(RETRY_STATUSES)}")
        print(f"Existing records: {len(existing_records)}")
    print("=" * 80)
    print()

    for index, instance in enumerate(selected_instances, start=1):
        instance_id = str(instance.get("row_id", "unknown"))
        existing_record = existing_records.get(instance["row_id"])

        if RESUME_RESULTS and should_skip_existing_record(existing_record):
            print(
                f"=== run_{run_number} | "
                f"Instance {index}/{len(selected_instances)}: {instance_id} "
                f"| skipping existing status={existing_record.get('status')} ==="
            )
            print()
            continue

        print(
            f"=== run_{run_number} | "
            f"Instance {index}/{len(selected_instances)}: {instance_id} ==="
        )

        record = run_instance(
            instance=instance,
            quality_aspect_descriptions=quality_aspect_descriptions,
            run_number=run_number,
            run_root_dir=run_root_dir,
            aggregated_results_file=aggregated_results_file,
        )

        if RESUME_RESULTS and record is not None:
            existing_records[record["id"]] = record
            write_jsonl_records_in_dataset_order(
                aggregated_results_file,
                existing_records,
                selected_instances,
            )

        print()

    print(f"Finished run_{run_number}.")
    print(f"Aggregated results file: {aggregated_results_file}")
    print()


def main() -> None:
    data = load_json(DATASET_FILE)
    quality_aspect_descriptions = load_quality_aspect_descriptions(QUALITY_ASPECTS_FILE)

    if NUM_INSTANCES_TO_RUN is None:
        selected_instances = data
    else:
        selected_instances = data[:NUM_INSTANCES_TO_RUN]

    print(f"Experiment name: {EXPERIMENT_NAME}")
    print(f"Experiment type: {EXPERIMENT_TYPE}")
    print(f"Experiment root: {EXPERIMENT_ROOT_DIR}")
    print(f"Prompt template: {PROMPT_TEMPLATE_FILE}")
    print(f"Dataset file: {DATASET_FILE}")
    print(f"Repos dir: {REPOS_DIR}")
    print(f"Running experiments dir: {RUNNING_EXPERIMENTS_DIR}")
    print(f"Lock dir: {LOCK_DIR}")
    print(f"Number of runs: {NUM_RUNS}")
    print(f"Instances per run: {len(selected_instances)}")
    print(f"Model provider: {MODEL_PROVIDER}")
    print(f"Model: {MODEL}")
    print(f"Correct answer: {CORRECT_ANSWER}")
    if RESUME_RESULTS:
        print(f"Resume mode: retry statuses {sorted(RETRY_STATUSES)}")

    for run_number in range(1, NUM_RUNS + 1):
        run_experiment_once(
            run_number=run_number,
            selected_instances=selected_instances,
            quality_aspect_descriptions=quality_aspect_descriptions,
        )

    print("Finished all runs.")
    print(f"Experiment root: {EXPERIMENT_ROOT_DIR}")


if __name__ == "__main__":
    main()
