# Projects & Experimentation Workflow

This directory contains our ML experimentation projects and defines our Machine Learning Developer Experience (ML DevEx) workflow.

## Overview

Our experimentation workflow follows a structured approach:

1. **Projects** - High-level goals or objectives we're working towards
2. **Experiments** - Incremental steps that move us toward project goals
3. **HEAD Loops** - Iterative cycles within each experiment

## Directory Structure

```
projects/
├── {ISSUE-NUMBER}-{description}/
│   ├── README.md                    # Project goal and context
│   └── experiments/
│       ├── YYYY-MM-DD-{experiment-name}.ipynb
│       └── YYYY-MM-DD-{experiment-name}.ipynb
```

### Example Structure

```
projects/
├── 1-dspy/
│   ├── README.md
│   └── experiments/
│       └── 2025-10-26-initial-pipeline.ipynb
```

## Projects

Projects represent high-level goals or objectives. Each project:

- Is named after its issue: `{ISSUE-NUMBER}-{description}` (matching the branch title)
- Contains a README.md describing the goal, motivation, and success criteria
- Has an `experiments/` directory for all related experimentation work

### Project Naming

Projects are named to match their corresponding issue and branch:

```
{ISSUE-NUMBER}-{description}
```

Examples:
- `1-dspy` (for issue #1 about DSPy integration)
- `42-optimize-retrieval` (for issue #42 about retrieval optimization)

## Experiments

Experiments are concrete, dated investigations that incrementally advance a project. Each experiment:

- Is a Jupyter notebook named with the date: `YYYY-MM-DD-{experiment-name}.ipynb`
- Lives in the project's `experiments/` directory
- Contains one or more HEAD loops
- Should call "production" CLI commands or the underlying `commands` module rather than reimplementing logic

### Experiment Naming

```
YYYY-MM-DD-{descriptive-experiment-name}.ipynb
```

Examples:
- `2025-10-26-initial-pipeline.ipynb`
- `2025-10-27-optimize-retrieval.ipynb`
- `2025-11-01-evaluate-llm-accuracy.ipynb`

## HEAD Loops

Each experiment contains multiple HEAD loops, where HEAD stands for:

- **Hypothesize** - Form a hypothesis about what might improve the system
- **Evaluate** - Run experiments to test the hypothesis
- **Analyze** - Examine the results and metrics
- **Decide** - Determine next steps based on the analysis

HEAD loops are the core iterative cycle for making progress within an experiment.

## MLflow Integration

We use MLflow for experiment tracking and model management.

### Experiment Naming Convention

MLflow experiments are named following a hierarchical pattern with forward slashes:

```
/{system}/{project}/{experiment}
```

Where:
- `system` - The top-level repository/system name
- `project` - The project identifier (e.g., `1-dspy`)
- `experiment` - The descriptive experiment name

The experiment name is automatically constructed from the `--project` and `--experiment-name` CLI arguments.

Example MLflow experiment names:
- `/symptom-diagnosis-explorer/1-dspy/initial-pipeline`
- `/symptom-diagnosis-explorer/1-dspy/optimize-retrieval`
- `/symptom-diagnosis-explorer/1-dspy/2025-10-26-hyperparameter-tuning`

### Usage

When running tuning commands, provide the `--project` and `--experiment-name` arguments:

```bash
symptom-diagnosis-explorer classify tune \
  --project 1-dspy \
  --experiment-name initial-pipeline
```

This creates an experiment named: `/symptom-diagnosis-explorer/1-dspy/initial-pipeline`

### Tagging

MLflow experiments are automatically tagged with:
- `system` - Hard-coded to `symptom-diagnosis-explorer`
- `project` - Set from the `--project` CLI argument

This enables filtering and organizing experiments across different projects and systems.

### Running MLflow

```bash
just mlflow
```

MLflow UI will be available at http://localhost:5000

## Jupyter Notebooks as Lab Notebooks

Jupyter notebooks in this workflow serve as **true lab notebooks**:

- Document the research process, hypotheses, and findings
- Should **call production CLI commands** or the underlying `commands` module
- Avoid reimplementing logic that exists in the codebase
- May occasionally include one-off standalone exploration notebooks (though this is unlikely)

### Best Practices

**DO:**
```python
# In notebook - call the production command
from symptom_diagnosis_explorer.commands.dataset import summary
result = summary.execute(dataset_name="train")
```

**DON'T:**
```python
# In notebook - reimplement logic
def calculate_summary(data):
    # Reimplementation of existing logic
    ...
```

## Workflow Summary

1. **Define a project** with clear goals in `projects/{ISSUE-NUMBER}-{description}/README.md`
2. **Create dated experiments** as Jupyter notebooks in the project's `experiments/` directory
3. **Run HEAD loops** within each experiment to iteratively make progress
4. **Track with MLflow** using consistent naming and tagging conventions
5. **Call production code** from notebooks to ensure reproducibility and maintainability

## Getting Started

To start a new experiment:

1. Ensure MLflow is running: `just mlflow`
2. Create a new dated notebook in your project's `experiments/` directory
3. Import and use the production CLI/commands in your notebook
4. Document your HEAD loops clearly
5. Log results to MLflow with appropriate tags
