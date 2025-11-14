# symptom-diagnosis-explorer

A CLI tool for exploring symptom diagnosis using DSPy-based classification models with MLFlow tracking.

## Dataset

This project uses the [symptom_to_diagnosis](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) dataset from Hugging Face, which contains patient symptoms and corresponding diagnoses.

## Installation

```bash
# Install dependencies using uv
uv sync

# Verify installation
uv run symptom-diagnosis-explorer --help
```

## Prerequisites

- Python 3.11+
- LLM provider access (choose one or more):

### LLM Provider Setup

This tool uses [DSPy](https://github.com/stanfordnlp/dspy) which supports multiple LLM providers. Choose the provider that works best for you:

#### OpenAI (Default)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Use with the CLI (default model)
uv run symptom-diagnosis-explorer classify tune

# Specify a different OpenAI model
uv run symptom-diagnosis-explorer classify tune --lm-model openai/gpt-4o
```

#### Anthropic (Claude)

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Use Claude Haiku with the CLI
uv run symptom-diagnosis-explorer classify tune --lm-model anthropic/claude-3-5-haiku-20241022
```

#### Ollama (Local, No API Key Required)

Ollama allows you to run models locally without any API keys or internet connection.

```bash
# 1. Install Ollama (https://ollama.ai)
# macOS/Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Start Ollama and pull a model
ollama pull llama3.1:8b

# 3. Use with the CLI (no API key needed!)
uv run symptom-diagnosis-explorer classify tune --lm-model ollama_chat/llama3.1:8b

# Other recommended Ollama models:
# - ollama_chat/llama3.1:70b (better quality, needs more RAM)
# - ollama_chat/mistral:7b (faster, lighter)
# - ollama_chat/qwen2.5:14b (good quality/speed balance)
```

## Usage

### Dataset Commands

Explore and analyze the symptom diagnosis dataset.

#### List Dataset Rows

```bash
# List first 5 rows from all data
uv run symptom-diagnosis-explorer dataset list

# List first 10 rows from training set
uv run symptom-diagnosis-explorer dataset list --split train --rows 10

# List rows from validation set
uv run symptom-diagnosis-explorer dataset list --split validation --rows 5

# List rows from test set
uv run symptom-diagnosis-explorer dataset list --split test
```

#### Dataset Summary Statistics

```bash
# Summary of all data
uv run symptom-diagnosis-explorer dataset summary

# Summary of training set
uv run symptom-diagnosis-explorer dataset summary --split train

# Summary of validation set
uv run symptom-diagnosis-explorer dataset summary --split validation

# Summary of test set
uv run symptom-diagnosis-explorer dataset summary --split test
```

### Classification Commands

Train, evaluate, and manage DSPy-based classification models with MLFlow tracking.

#### Tune/Train a Model

Optimize a classification model using DSPy optimizers with automatic validation evaluation.

```bash
# Basic tuning with defaults (BootstrapFewShot optimizer, gpt-4o-mini)
uv run symptom-diagnosis-explorer classify tune

# Use with Ollama (local, no API key needed)
uv run symptom-diagnosis-explorer classify tune \
  --lm-model ollama_chat/llama3.1:8b

# Use with Claude
uv run symptom-diagnosis-explorer classify tune \
  --lm-model anthropic/claude-3-5-haiku-20241022

# Use MIPROv2 optimizer with limited dataset
uv run symptom-diagnosis-explorer classify tune \
  --optimizer mipro \
  --train-size 100 \
  --val-size 20

# Use different OpenAI model
uv run symptom-diagnosis-explorer classify tune \
  --lm-model openai/gpt-4o

# Custom model name and experiment using project and experiment name
uv run symptom-diagnosis-explorer classify tune \
  --model-name my-classifier \
  --project my-project \
  --experiment-name my-experiment

# Adjust optimizer parameters
uv run symptom-diagnosis-explorer classify tune \
  --num-threads 8 \
  --max-bootstrapped-demos 5 \
  --max-labeled-demos 6
```

**Tune Command Options:**
- `--optimizer`: Optimizer type (`bootstrap` or `mipro`, default: `bootstrap`)
- `--train-size`: Limit training examples (default: all)
- `--val-size`: Limit validation examples (default: all)
- `--model-name`: Model name for MLFlow registry (default: `symptom-classifier`)
- `--project`: Project identifier for MLFlow (required, e.g., `1-dspy`)
- `--experiment-name`: Experiment name for MLFlow (required, auto-constructs `/symptom-diagnosis-explorer/{project}/{experiment-name}`)
- `--lm-model`: Language model identifier (default: `openai/gpt-4o-mini`)
- `--num-threads`: Number of parallel threads (default: `4`)
- `--max-bootstrapped-demos`: Max bootstrapped demonstrations (default: `3`)
- `--max-labeled-demos`: Max labeled demonstrations (default: `4`)

#### Evaluate a Model

Evaluate a saved model on a dataset split.

```bash
# Evaluate latest model on test set (default)
uv run symptom-diagnosis-explorer classify evaluate

# Evaluate on validation set
uv run symptom-diagnosis-explorer classify evaluate --split validation

# Evaluate specific model version
uv run symptom-diagnosis-explorer classify evaluate \
  --model-name symptom-classifier \
  --model-version 2 \
  --split test

# Evaluate on training set
uv run symptom-diagnosis-explorer classify evaluate --split train
```

**Evaluate Command Options:**
- `--model-name`: Model name in registry (default: `symptom-classifier`)
- `--model-version`: Model version (default: latest)
- `--split`: Dataset split (`train`, `validation`, or `test`, default: `test`)

#### List Registered Models

List models in the MLFlow registry.

```bash
# List all registered models
uv run symptom-diagnosis-explorer classify list-models

# Filter by name
uv run symptom-diagnosis-explorer classify list-models --name-filter symptom
```

**List Models Command Options:**
- `--name-filter`: Filter models by name (substring match)

## MLFlow Tracking

This project uses MLFlow for experiment tracking and model registry.

### Local MLFlow Setup

By default, MLFlow tracking data is stored in the `.mlflow` directory. You can view the MLFlow UI:

```bash
# Start MLFlow UI server
uv run mlflow ui --backend-store-uri file:///.mlflow

# Open in browser: http://localhost:5000
```

### MLFlow Run Organization

- **Experiments**: Organized hierarchically as `/symptom-diagnosis-explorer/{project}/{experiment-name}` (e.g., `/symptom-diagnosis-explorer/1-dspy/initial-pipeline`)
- **Runs**: Individual tuning/evaluation executions with metrics and artifacts
- **Models**: Registered models with versions and aliases
- **Artifacts**: Logged files including prediction samples and disagreements

For more details on the naming convention, see [projects/README.md](projects/README.md#experiment-naming-convention).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup, workflow, and quality expectations for contributors.

### Running Tests

```bash
# Run all tests
just test

# Run with coverage
just test-cov
```

### Project Structure

```
symptom-diagnosis-explorer/
├── src/symptom_diagnosis_explorer/
│   ├── cli.py                  # CLI entry point
│   ├── commands/               # Command layer (request/response models)
│   │   ├── dataset/           # Dataset commands
│   │   └── classify/          # Classification commands
│   ├── services/              # Service layer (business logic)
│   │   ├── dataset.py         # Dataset loading and management
│   │   └── model_development.py  # DSPy model training and evaluation
│   └── models/                # Domain models and schemas
│       ├── domain.py          # Core domain models
│       ├── dataset.py         # Dataset schemas
│       └── model_development.py  # Model configuration and metrics
└── tests/
    └── integration/           # Integration tests
```

## License

MIT
