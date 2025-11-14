# Contributing to Symptom Diagnosis Explorer

## Development Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) - Python package and project manager
- Just - Command runner (install via `brew install just` on macOS)

### Installation

Install dependencies:

```bash
just install
```

## Development Workflow

### Running Tests

```bash
# Run all tests
just test

# Run specific test suites
just test-unit
just test-integration
just test-e2e
```

The `just` targets are the recommended defaults, but feel free to invoke the underlying tools directly when you need more control:

```bash
uv run pytest tests/unit -k tokenizer
uv run pytest tests/integration --maxfail=1
uv run pytest tests/e2e -n auto
```

### Code Quality

```bash
# Run linter
just lint

# Auto-fix linting issues
just lint-fix

# Format code
just format

# Check formatting
just format-check

# Type checking
just typecheck

# Run all CI checks
just ci
```

You can also copy the arguments from the `justfile` and run the tools directly:

```bash
uv run ruff check src --fix
uv run ruff format
uv run mypy src
```

### Jupyter Notebooks

```bash
# Create kernel for this environment
just kernel

# Launch Jupyter Lab
just notebook
```

### MLflow

MLflow is used for experiment tracking and model management. The MLflow UI can be launched locally using:

```bash
just mlflow
```

This will:
- Start the MLflow UI on http://localhost:5000 (default)
- Store experiment data in the `.mlflow` directory (gitignored)
- Store artifacts in `.mlflow/artifacts`

The `.mlflow` directory is gitignored and stored locally on your filesystem. In the future, this may be replaced with a standalone MLflow instance.

## Experimentation Workflow

For ML experimentation and the ML DevEx workflow, see [projects/README.md](projects/README.md).

This includes:
- Project structure and naming conventions
- Experiment organization and HEAD loops (Hypothesize, Evaluate, Analyze, Decide)
- MLflow integration and naming conventions
- Best practices for using Jupyter notebooks as lab notebooks
