# QualDeltaBench: A Review-Grounded Benchmark for Evaluating LLMs as Code Quality Judges

This repository contains the datasets, mining scripts, manual validation materials, prompts, and experiment runners used in our study.

The package is organized to support the full pipeline: from mining reviewer comments and follow-up commits, to extracting function/file-level instances, to running the experiments in different scopes, quality-aspect settings, programming languages, and models.

## Repository Structure

```
.
├── datasets/
│   ├── java_dataset.json.zip
│   └── python_dataset.json.zip
│
├── manual_classification_and_validation/
│   ├── manually_validated_instances_java.xlsx
│   ├── manually_validated_instances_python.xlsx
│   ├── prompt_atomic_change.txt
│   └── prompt_classification.txt
│
├── mining/
│   ├── 500_random_java_repos.csv
│   ├── 500_random_python_repos.csv
│   ├── extract_functions_before_after_file_and_function_level.py
│   ├── mining_commit_after.py
│   └── mining_review_comments_from_repos.py
│
└── running_experiments/
    ├── prompts/
    │   ├── prompt_agentic.txt
    │   ├── prompt_agentic_without_quality_aspects.txt
    │   ├── prompt_before_only_agentic.txt
    │   ├── prompt_before_only_agentic_without_quality_aspects.txt
    │   ├── prompt_file.txt
    │   ├── prompt_file_without_quality_aspects.txt
    │   ├── prompt_function.txt
    │   ├── prompt_function_without_quality_aspects.txt
    │   ├── prompt_one_file.txt
    │   ├── prompt_one_file_without_quality_aspects.txt
    │   ├── prompt_one_function.txt
    │   └── prompt_one_function_without_quality_aspects.txt
    ├── quality_aspects.txt
    ├── run_agent.py
    ├── run_agent_before_only.py
    ├── run_llm_function_file.py
    └── run_llm_function_file_before_only.py
```

## Datasets

The `datasets/` directory contains the final datasets obtained after mining, automatic filtering, and manual validation.

- `java_dataset.json.zip`: final Java dataset.
- `python_dataset.json.zip`: final Python dataset.

These datasets contain the validated instances used in the experiments.

## Manual Classification and Validation

The `manual_classification_and_validation/` directory contains the materials used for automatic classification and manual validation.

- `prompt_classification.txt`: prompt used to classify reviewer comments according to the defined quality attributes.
- `prompt_atomic_change.txt`: prompt used to check whether the code change associated with a reviewer comment is atomic.
- `manually_validated_instances_java.xlsx`: final manually validated Java instances.
- `manually_validated_instances_python.xlsx`: final manually validated Python instances.

The validated Excel files contain the instances that passed the manual validation step and were used to build the final datasets.

## Mining Pipeline

The `mining/` directory contains the scripts used to collect and process reviewer comments and the corresponding code changes.

### 1. Mining Reviewer Comments

```text
mining_review_comments_from_repos.py
```

This script mines pull request review comments from GitHub repositories.

The input repository lists are:

```text
500_random_java_repos.csv
500_random_python_repos.csv
```

Each list contains 500 randomly selected repositories for the corresponding programming language.

The script collects review comments and saves information such as the commented file, the pull request, the comment body, the diff hunk, and the file content at the commit where the review comment was made.

### 2. Mining Follow-up Commits

```text
mining_commit_after.py
```

This script searches for a later commit in the same pull request that may address a previously mined reviewer comment.

For each mined review comment, the script looks for a subsequent commit that modifies the same portion of the same file that was commented on. The resulting commit is referred to as the `commit after`.

This step is used to identify candidate code changes that may implement or address the reviewer comment.

### 3. Extracting Function-Level and File-Level Instances

```text
extract_functions_before_after_file_and_function_level.py
```

This script localizes the function or method associated with each reviewer comment, when such a function exists.

For each valid instance, the script extracts:

- the function before the follow-up commit;
- the function after the follow-up commit;
- the full file before the follow-up commit;
- the full file after the follow-up commit.

Some reviewer comments may refer to code outside a function or method. In those cases, function-level extraction may fail or be unavailable.

## Running Experiments

The `running_experiments/` directory contains the scripts and prompts used to run the experiments.

The experiments are organized along three scopes, referred to in the paper as:

- **function-scoped** experiments;
- **file-scoped** experiments;
- **repository-scoped** experiments.

The repository-scoped setting is implemented with an agentic workflow.

The experiments can also be run in two quality-aspect settings:

- **aspect-conditioned**: the model receives the relevant quality aspect information;
- **open-aspect**: the model does not receive explicit quality aspect information.

The scripts support both Java and Python datasets and can be used with different models, including GPT, MiniMax, and Gemini.

### Experiment Scripts

```text
run_llm_function_file.py
```

Runs the standard non-agentic experiments for the function-scoped and file-scoped settings (**comparative judging**).

```text
run_llm_function_file_before_only.py
```

Runs the **single-version issue diagnosis** version of the non-agentic experiments. In this setting, the model receives only the code before the change.

```text
run_agent.py
```

Runs the repository-scoped agentic experiments (**comparative judging**).

```text
run_agent_before_only.py
```

Runs the **ingle-version issue diagnosis** version of the repository-scoped agentic experiments.

### Prompts

The `running_experiments/prompts/` directory contains the prompts used for each experimental setting.

Repository-scoped prompts:

```text
prompt_agentic.txt
prompt_agentic_without_quality_aspects.txt
prompt_before_only_agentic.txt
prompt_before_only_agentic_without_quality_aspects.txt
```

File-scoped prompts:

```text
prompt_file.txt
prompt_file_without_quality_aspects.txt
prompt_one_file.txt
prompt_one_file_without_quality_aspects.txt
```

Function-scoped prompts:

```text
prompt_function.txt
prompt_function_without_quality_aspects.txt
prompt_one_function.txt
prompt_one_function_without_quality_aspects.txt
```

The `quality_aspects.txt` file contains the list of quality attributes used in the aspect-conditioned experiments.

## Configuration

The scripts are configured through constants defined at the top of each file.

Typical configuration options include:

```python
LANGUAGE = "python"  # Options: "python", "java"
EXPERIMENT_TYPE = "with_qa"  # Options: "with_qa", "without_qa" (aspect-conditioned/open-aspect)
MODEL_PROVIDER = "openai"  # Options: "openai", "openrouter"
MODEL = "openai/gpt-5.4-mini"
```

For OpenRouter-based runs, use for example:

```python
MODEL_PROVIDER = "openrouter"
MODEL = "minimax/minimax-m3"
```

or:

```python
MODEL_PROVIDER = "openrouter"
MODEL = "google/gemini-2.5-pro"
```

## API Keys

For OpenAI, set:

```text
OPENAI_API_KEY=your_key_here
```

For OpenRouter, set:

```text
OPENROUTER_API_KEY=your_key_here
```

Keys can be provided through the environment or through a local `.env` file, depending on the script configuration.

## Expected Folder Layout for Experiments

The experiment scripts assume that `running_experiments/` contains the experiment data, repositories, prompts, and configuration files required to execute the runs.

A typical layout is:

```text
running_experiments/
├── data/
│   ├── java_dataset.json
│   └── python_dataset.json
├── repos_java/
├── repos_python
├── prompts/
├── quality_aspects.txt
├── run_agent.py
├── run_agent_before_only.py
├── run_llm_function_file.py
└── run_llm_function_file_before_only.py
```

The `repos_java/` and `repos_python/` directories contain the local repository snapshots used by the agentic experiments.

## Reproducing the Pipeline

A high-level reproduction workflow is:

1. Mine review comments from the selected repositories:

```bash
python mining/mining_review_comments_from_repos.py
```

2. Mine follow-up commits that may address the review comments:

```bash
python mining/mining_commit_after.py
```

3. Extract function-level and file-level before/after code:

```bash
python mining/extract_functions_before_after_file_and_function_level.py
```

4. Use the classification prompts and manual validation files in:

```text
manual_classification_and_validation/
```

5. Run the desired experiments from:

```text
running_experiments/
```

For example:

```bash
python running_experiments/run_llm_function_file.py
python running_experiments/run_agent.py
```

Before running each script, check the constants at the top of the file to select the language, task setting, scope, model provider, and model.
