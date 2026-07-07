#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# The script expects this directory to contain:
# - data/java_dataset.json
# - data/python_dataset.json
# - prompts/prompt_one_function.txt
# - prompts/prompt_one_function_without_quality_aspects.txt
# - prompts/prompt_one_file.txt
# - prompts/prompt_one_file_without_quality_aspects.txt
# - quality_aspects.txt
# - optional .env file with OPENAI_API_KEY and/or OPENROUTER_API_KEY
# - optional key files: openai_apikey.txt, openrouter_token.txt, openrouter_apikey.txt
RUNNING_EXPERIMENTS_DIR = SCRIPT_DIR / "running_experimens"

# Options: "java", "python", or "all".
LANGUAGE = "python"

# Options: "function", "file", or "all".
EXPERIMENT_LEVEL = "file"

# Options:
# - "with_quality_aspects": the model receives the target quality aspect.
# - "without_quality_aspects": the model receives all quality aspects and must select the relevant one(s).
# - "all": run both settings.
EXPERIMENT_TYPE = "without_quality_aspects"

# Options: "openai", "openrouter".
MODEL_PROVIDER = "openai"

# Examples:
# - OpenAI direct: "gpt-5.4-mini" or "openai/gpt-5.4-mini"
# - OpenRouter:   "minimax/minimax-m3"
# - OpenRouter:   "google/gemini-2.5-pro"
MODEL = "gpt-5.4-mini"

# Number of repetitions for each selected configuration.
N_RUNS = 10

# None = process all rows. Use an integer for quick tests.
LIMIT: int | None = None

# None = no explicit seeding. Integer = reproducible order per run.
RANDOM_SEED: int | None = None

# If True, overwrite existing JSONL files for the same run.
OVERWRITE_OUTPUT = True

SYSTEM_PROMPT = "You are an expert code judge."
TEMPERATURE: float | None = None
CORRECT_ANSWER = "Yes"

OUTPUT_DIR = RUNNING_EXPERIMENTS_DIR / "simple_model_outputs"


# ============================================================
# Paths and validation
# ============================================================

load_dotenv(RUNNING_EXPERIMENTS_DIR / ".env")

DATASET_FILES = {
    "java": RUNNING_EXPERIMENTS_DIR / "data" / "java_dataset.json",
    "python": RUNNING_EXPERIMENTS_DIR / "data" / "python_dataset.json",
}

PROMPT_FILES = {
    ("function", "with_quality_aspects"): RUNNING_EXPERIMENTS_DIR / "prompts" / "prompt_one_function.txt",
    ("function", "without_quality_aspects"): RUNNING_EXPERIMENTS_DIR / "prompts" / "prompt_one_function_without_quality_aspects.txt",
    ("file", "with_quality_aspects"): RUNNING_EXPERIMENTS_DIR / "prompts" / "prompt_one_file.txt",
    ("file", "without_quality_aspects"): RUNNING_EXPERIMENTS_DIR / "prompts" / "prompt_one_file_without_quality_aspects.txt",
}

QUALITY_ASPECTS_FILE = RUNNING_EXPERIMENTS_DIR / "quality_aspects.txt"

FIELD_ALIASES = {
    "function_before": ["method_before", "function_before"],
    "file_before": ["file_before", "file_content_at_comment_commit"],
}

VALID_LANGUAGES = set(DATASET_FILES)
VALID_LEVELS = {"function", "file"}
VALID_TYPES = {"with_quality_aspects", "without_quality_aspects"}

OPENAI_KEY_FILES = [RUNNING_EXPERIMENTS_DIR / "openai_apikey.txt"]
OPENROUTER_KEY_FILES = [
    RUNNING_EXPERIMENTS_DIR / "openrouter_token.txt",
    RUNNING_EXPERIMENTS_DIR / "openrouter_apikey.txt",
]


# ============================================================
# Generic helpers
# ============================================================

def expand_config(value: str, valid_values: set[str], field_name: str) -> list[str]:
    normalized = value.lower().strip()
    if normalized == "all":
        return sorted(valid_values)
    if normalized not in valid_values:
        raise ValueError(
            f"Unsupported {field_name}={value!r}. "
            f"Supported values: {sorted(valid_values)} or 'all'."
        )
    return [normalized]


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as input_file:
        return "\n".join(line.strip() for line in input_file)


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON list: {path}")
    return data


def load_quality_aspects(path: Path = QUALITY_ASPECTS_FILE) -> dict[str, str]:
    aspects: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing quality aspects file: {path}")

    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            key, description = line.split(" -- ", 1)
            description = description.replace("Characterized by concepts like:", "").strip()
            aspects[key] = description
    return aspects


def first_existing_value(row: dict[str, Any], aliases: list[str], *, row_id: Any) -> str:
    for name in aliases:
        value = row.get(name)
        if value is not None:
            return str(value)
    raise KeyError(f"Row {row_id!r} is missing all expected fields: {aliases}")


def row_identifier(row: dict[str, Any]) -> Any:
    return row.get("row_id", row.get("id", row.get("review_comment_id")))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()


def slugify_for_path(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def output_model_name(model: str) -> str:
    lowered = model.lower()
    if "minimax" in lowered:
        return "minimax"
    if "gemini" in lowered:
        return "gemini"
    if "gpt" in lowered or "openai" in lowered:
        return "gpt"
    return slugify_for_path(model)


# ============================================================
# API client
# ============================================================

def read_key_from_files(paths: list[Path]) -> str | None:
    for path in paths:
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
    return None


def load_api_key(provider: str) -> str:
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY") or read_key_from_files(OPENAI_KEY_FILES)
        if not key:
            raise RuntimeError("Missing OpenAI key. Set OPENAI_API_KEY or create openai_apikey.txt.")
        return key

    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY") or read_key_from_files(OPENROUTER_KEY_FILES)
        if not key:
            raise RuntimeError(
                "Missing OpenRouter key. Set OPENROUTER_API_KEY or create openrouter_token.txt."
            )
        return key

    raise ValueError("MODEL_PROVIDER must be either 'openai' or 'openrouter'.")


def model_name_for_provider(provider: str, model: str) -> str:
    if provider == "openai" and model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def create_client(provider: str) -> OpenAI:
    api_key = load_api_key(provider)
    if provider == "openai":
        return OpenAI(api_key=api_key)

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"X-Title": "Replication package before-only simple model experiment"},
    )


CLIENT = create_client(MODEL_PROVIDER)
API_MODEL = model_name_for_provider(MODEL_PROVIDER, MODEL)


def ask_model(prompt: str) -> str:
    request: dict[str, Any] = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if TEMPERATURE is not None:
        request["temperature"] = TEMPERATURE

    response = CLIENT.chat.completions.create(**request)
    return response.choices[0].message.content or ""


# ============================================================
# Response parsing
# ============================================================

def parse_bullet_section(text: str, section_name: str, next_sections: list[str]) -> list[str]:
    next_sections_pattern = "|".join(re.escape(section) for section in next_sections)
    pattern = rf"{re.escape(section_name)}\s*:\s*(.*?)(?=\n\s*(?:{next_sections_pattern})\s*:|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []

    items: list[str] = []
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"^\d+\.\s*", "", line)
        if line and line.lower() != "none":
            items.append(line)
    return items


def parse_text_section(text: str, section_name: str, next_sections: list[str]) -> str | None:
    next_sections_pattern = "|".join(re.escape(section) for section in next_sections)
    pattern = rf"{re.escape(section_name)}\s*:\s*(.*?)(?=\n\s*(?:{next_sections_pattern})\s*:|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    value = match.group(1).strip()
    return value if value and value.lower() != "none" else None


def parse_model_response(response: str, include_quality_aspects: bool) -> dict[str, Any]:
    issue_match = re.search(
        r"^\s*Issue\s*:\s*(Yes|No)\s*$",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    parsed: dict[str, Any] = {
        "decision": issue_match.group(1).capitalize() if issue_match else None,
        "key_evidence": parse_text_section(
            response,
            section_name="Evidence",
            next_sections=["Issue", "Quality aspects"],
        ),
    }

    if include_quality_aspects:
        parsed["quality_aspects"] = parse_bullet_section(
            response,
            section_name="Quality aspects",
            next_sections=["Issue", "Evidence"],
        )

    return parsed


# ============================================================
# Prompting and output
# ============================================================

def build_prompt(
    row: dict[str, Any],
    prompt_template: str,
    language: str,
    experiment_level: str,
    experiment_type: str,
    quality_aspects: dict[str, str] | None = None,
) -> str:
    rid = row_identifier(row)

    if experiment_level == "function":
        implementation = first_existing_value(row, FIELD_ALIASES["function_before"], row_id=rid)
        placeholder = "{{IMPLEMENTATION}}"
    else:
        implementation = first_existing_value(row, FIELD_ALIASES["file_before"], row_id=rid)
        placeholder = "{{FILE}}"

    prompt = (
        prompt_template
        .replace(placeholder, implementation)
        .replace("{{LANGUAGE}}", language.capitalize())
    )

    if experiment_level == "file":
        prompt = prompt.replace("{{TARGET_FUNCTION_SIGNATURE}}", str(row.get("signature", "")))

    if experiment_type == "with_quality_aspects":
        if quality_aspects is None:
            raise ValueError("Quality aspects are required for with_quality_aspects experiments.")
        aspect_name = str(row["classification"])
        if aspect_name not in quality_aspects:
            raise KeyError(f"Unknown quality aspect {aspect_name!r} for row {rid!r}.")
        prompt = (
            prompt
            .replace("{{QUALITY_ASPECT_NAME}}", aspect_name)
            .replace("{{QUALITY_ASPECT_DESCRIPTION}}", quality_aspects[aspect_name])
        )

    return prompt


def experiment_type_folder(experiment_type: str) -> str:
    return {
        "with_quality_aspects": "before_only_with_qa",
        "without_quality_aspects": "before_only_without_qa",
    }[experiment_type]


def make_output_dir(language: str, experiment_level: str, experiment_type: str) -> Path:
    return (
        OUTPUT_DIR
        / output_model_name(MODEL)
        / language
        / f"{experiment_level}_level"
        / experiment_type_folder(experiment_type)
    )


def make_output_file(
    language: str,
    experiment_level: str,
    experiment_type: str,
    run_id: int,
) -> Path:
    return make_output_dir(language, experiment_level, experiment_type) / f"model_outputs_before_only_run_{run_id:02d}.jsonl"


def run_experiment(
    language: str,
    experiment_level: str,
    experiment_type: str,
    run_id: int,
    limit: int | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    if random_seed is not None:
        random.seed(random_seed + run_id)

    data = load_json(DATASET_FILES[language])
    prompt_template = read_text(PROMPT_FILES[(experiment_level, experiment_type)])
    quality_aspects = load_quality_aspects() if experiment_type == "with_quality_aspects" else None

    output_file = make_output_file(language, experiment_level, experiment_type, run_id)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if OVERWRITE_OUTPUT and output_file.exists():
        output_file.unlink()

    n_correct = 0
    n_total = 0
    rows_to_process = data[:limit] if limit is not None else data

    for row in rows_to_process:
        prompt = build_prompt(
            row=row,
            prompt_template=prompt_template,
            language=language,
            experiment_level=experiment_level,
            experiment_type=experiment_type,
            quality_aspects=quality_aspects,
        )

        response = ask_model(prompt)
        parsed_response = parse_model_response(
            response,
            include_quality_aspects=(experiment_type == "without_quality_aspects"),
        )

        decision = parsed_response["decision"]
        is_correct_answer = (decision or "").lower() == CORRECT_ANSWER.lower()
        n_correct += int(is_correct_answer)
        n_total += 1

        record = {
            "id": row_identifier(row),
            "run_id": run_id,
            "language": language,
            "experiment_level": experiment_level,
            "experiment_type": experiment_type,
            "model_provider": MODEL_PROVIDER,
            "model": MODEL,
            "classification": row.get("classification"),
            "prompt": prompt,
            "model_response_raw": response,
            "decision": decision,
            "key_evidence": parsed_response["key_evidence"],
            "correct_answer": CORRECT_ANSWER,
            "is_correct_answer": is_correct_answer,
        }

        if experiment_type == "without_quality_aspects":
            record["quality_aspects"] = parsed_response.get("quality_aspects", [])

        append_jsonl(output_file, record)

    accuracy = n_correct / n_total if n_total else 0.0
    summary = {
        "language": language,
        "experiment_level": experiment_level,
        "experiment_type": experiment_type,
        "run_id": run_id,
        "model_provider": MODEL_PROVIDER,
        "model": MODEL,
        "output_file": str(output_file),
        "correct": n_correct,
        "total": n_total,
        "accuracy": accuracy,
    }

    print(
        f"[{language} | {experiment_level} | {experiment_type} | run {run_id:02d}/{N_RUNS:02d}] "
        f"Correct: {n_correct}/{n_total} - Accuracy: {accuracy:.2%} - Output: {output_file}"
    )
    return summary


def main() -> None:
    languages = expand_config(LANGUAGE, VALID_LANGUAGES, "LANGUAGE")
    experiment_levels = expand_config(EXPERIMENT_LEVEL, VALID_LEVELS, "EXPERIMENT_LEVEL")
    experiment_types = expand_config(EXPERIMENT_TYPE, VALID_TYPES, "EXPERIMENT_TYPE")

    summaries: list[dict[str, Any]] = []
    for language in languages:
        for experiment_level in experiment_levels:
            for experiment_type in experiment_types:
                for run_id in range(1, N_RUNS + 1):
                    summaries.append(
                        run_experiment(
                            language=language,
                            experiment_level=experiment_level,
                            experiment_type=experiment_type,
                            run_id=run_id,
                            limit=LIMIT,
                            random_seed=RANDOM_SEED,
                        )
                    )

    summary_file = OUTPUT_DIR / output_model_name(MODEL) / "before_only_experiment_summaries.jsonl"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as output_file:
        for summary in summaries:
            output_file.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
