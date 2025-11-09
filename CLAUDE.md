# Autonomous Coding Agent Guidelines

This document contains guidance for any code-generation agent collaborating on this project.

Please also review the repository documentation for additional context:
- [README.md](README.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- Additional references in the `docs/` directory

## Commit Messages

Use the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for all commit messages.
Example conventional commit: `feat: add dataset summary command`.
Keep commit bodies concise and focused on what changed and why.

## Testing

Prefer the recipes in the project's `justfile` for common workflows, but feel free to run the underlying tooling directly via `uv run ...` commands whenever you need custom flags or tighter scopes.

```bash
# Reproducible defaults from the justfile
just test
just test-integration

# Direct invocations for targeted debugging
uv run pytest tests/unit -k tokenizer
uv run pytest tests/integration --maxfail=1
```

## Quality Checks

Use the commands in `justfile` as a quick reference for linters, formatters, and type checkers. Coding agents can copy those options or run the tools directly.

```bash
# Recipes
just lint
just format

# Direct tooling
uv run ruff check src --fix
uv run ruff format
uv run mypy src
```

## Pull Requests

- Keep changes tightly scoped; split work into separate PRs when needed.
- Summarize modifications and corresponding tests in the final hand-off message.
- Reference relevant issues or tickets in the PR description when applicable.

## Project Structure

This project follows a layered architecture:

```
CLI Layer (cli.py)
    ↓
Commands Layer (commands/)
    ↓
Services Layer (services/)
    ↓
Models Layer (models/)
    ↓
External Dependencies (DSPy, LangChain, MLFlow, HuggingFace Datasets)
```

### Multi-Framework Architecture

The classification system supports multiple ML frameworks through a registry pattern:

```
services/ml_models/
├── __init__.py           # Public API exports
├── base.py               # BaseModelService abstract class
├── registry.py           # FrameworkRegistry for service discovery
├── dspy.py               # DSPyModelService (trainable)
└── langchain.py          # LangChainModelService (non-trainable)
```

**Key Components:**

1. **BaseModelService** (`base.py`): Abstract base class defining the interface all frameworks must implement
   - Abstract methods: `tune()`, `evaluate()`
   - Abstract properties: `requires_training`, `framework_type`
   - Shared helpers: `_log_dataset_info()`, `_setup_mlflow_experiment()`

2. **FrameworkRegistry** (`registry.py`): Centralized registry for framework service classes
   - Services self-register using `@FrameworkRegistry.register(FrameworkType.FRAMEWORK_NAME)`
   - Factory method: `create_service(config, dataset_service)` instantiates the correct service
   - Discovery: `list_frameworks()` returns available frameworks

3. **Framework Services**: Concrete implementations
   - **DSPyModelService**: Trainable models using DSPy optimizers (BootstrapFewShot, MIPROv2)
   - **LangChainModelService**: Non-trainable models using prompt engineering with LCEL chains
   - **PydanticAIModelService**: Non-trainable models using agent-based prompting with Pydantic validation

**Framework Characteristics:**

| Framework   | Requires Training | MLFlow Usage | Evaluation Strategy |
|-------------|-------------------|--------------|---------------------|
| DSPy        | Yes               | Loads compiled model from MLFlow | Uses saved optimization artifacts |
| LangChain   | No                | Only logs metrics/params | Recreates chain from hardcoded prompts |
| Pydantic-AI | No                | Only logs metrics/params | Recreates agent from config |

### Adding a New Framework

To add support for a new ML framework, follow these steps:

**1. Create Framework Configuration Model** (`models/model_development.py`):
```python
from pydantic import BaseModel, Field, computed_field
from typing import Literal

class NewFrameworkConfig(BaseFrameworkConfig):
    """Configuration for NewFramework."""
    framework: Literal[FrameworkType.NEW_FRAMEWORK] = FrameworkType.NEW_FRAMEWORK

    # Framework-specific settings
    some_setting: str = Field(description="Framework-specific setting")

    @computed_field
    @property
    def requires_training(self) -> bool:
        return True  # or False, depending on framework
```

**2. Add to FrameworkType Enum**:
```python
class FrameworkType(str, Enum):
    DSPY = "dspy"
    LANGCHAIN = "langchain"
    NEW_FRAMEWORK = "new_framework"  # Add this
```

**3. Update FrameworkConfig Union**:
```python
FrameworkConfig = Annotated[
    Union[DSPyConfig, LangChainConfig, NewFrameworkConfig],  # Add NewFrameworkConfig
    Field(discriminator="framework")
]
```

**4. Create Service Implementation** (`services/ml_models/new_framework.py`):
```python
from services.ml_models.base import BaseModelService
from services.ml_models.registry import FrameworkRegistry
from models.model_development import FrameworkType

@FrameworkRegistry.register(FrameworkType.NEW_FRAMEWORK)
class NewFrameworkModelService(BaseModelService):
    """Service for NewFramework-based models."""

    @property
    def framework_type(self) -> str:
        return FrameworkType.NEW_FRAMEWORK

    @property
    def requires_training(self) -> bool:
        return True  # or False

    def tune(self, train_size, val_size, model_name) -> tuple[TuneMetrics, ModelInfo]:
        # Implement training/tuning logic
        pass

    def evaluate(self, model_name, model_version, split, eval_size) -> EvaluateMetrics:
        # Implement evaluation logic
        pass
```

**5. Update Command Layer** (`commands/classify/tune.py` and `evaluate.py`):
```python
# In TuneCommand.__init__:
elif request.framework == FrameworkType.NEW_FRAMEWORK:
    framework_config = NewFrameworkConfig(
        some_setting=request.new_framework_some_setting,
    )
```

**6. Update CLI** (`cli.py`):
```python
# Add framework-specific CLI options with prefix
new_framework_some_setting: Annotated[
    str,
    typer.Option(help="[NewFramework] Description of setting"),
] = "default_value"
```

**7. Add Tests** (`tests/integration/services/ml_models/test_new_framework_integration.py`):
```python
@pytest.mark.integration
@pytest.mark.llm
class TestNewFrameworkModelService:
    def test_tune_with_small_dataset(self, ...):
        # Test tuning workflow
        pass

    def test_evaluate_on_test_split(self, ...):
        # Test evaluation workflow
        pass
```

**8. Update Exports** (`services/ml_models/__init__.py`):
```python
from symptom_diagnosis_explorer.services.ml_models.new_framework import NewFrameworkModelService

__all__ = [
    "BaseModelService",
    "FrameworkRegistry",
    "DSPyModelService",
    "LangChainModelService",
    "NewFrameworkModelService",  # Add this
]
```

## Code Patterns

### Imports
- Use top-level, absolute imports only
- NO guarded imports (imports inside functions)
- All imports should be at the module level

### Services
- Use lazy loading and caching for expensive operations
- Store cached data in private attributes (prefixed with `_`)
- Raise `RuntimeError` if dependencies aren't loaded yet
- Follow existing patterns in `services/dataset.py`

### Commands
- Use Pydantic models for request/response validation
- Commands delegate to services (no business logic in commands)
- Follow existing patterns in `commands/dataset/`

### Models
- Use Pydantic for data validation
- Use Pandera for DataFrame schema validation
- Type hints throughout

## Dataset Splits

The dataset has the following splits:
- `train`: Training data (after 80/20 split: ~853 rows)
- `validation`: Validation data (after 80/20 split: ~171 rows)
- `test`: Test data (original test split)
- `all`: All data concatenated

The train/validation split is deterministic (random_state=42) and cached to ensure consistency.

## Development Workflow

1. Make changes to code
2. Run tests via `just test` or direct commands such as `uv run pytest tests/unit -k my_case`
3. Verify via CLI commands (e.g., `symptom-diagnosis-explorer dataset summary --split validation`)
4. Check types/linting as needed using `just lint`, `uv run ruff check`, `uv run mypy`, etc.
