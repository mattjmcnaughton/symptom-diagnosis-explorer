# Research Execution Document

## Executive Summary

This document contains executed research findings for implementing a multi-framework classification architecture that supports both DSPy and LangChain LCEL. Research covered current codebase structure, LangChain structured output patterns, MLFlow integration strategies, Pydantic discriminated unions, and registry patterns.

**Key Finding:** The current codebase has a well-structured DSPy implementation with comprehensive MLFlow tracking. The new LangChain implementation should follow similar patterns for consistency, using discriminated unions for framework selection and a registry pattern for service instantiation.

---

## Phase 1: Current Codebase Analysis

### Current Service Structure (services/model_development.py)

**Location:** `src/symptom_diagnosis_explorer/services/model_development.py` (899 lines)

**Key Methods:**
- `__init__(config: ClassificationConfig)` - Initializes DSPy LM and MLFlow tracking
- `tune(train_size, val_size, model_name)` - Main training method, returns `TuneMetrics` and `ModelInfo`
- `evaluate(model_name, model_version, split, eval_size)` - Returns `EvaluateMetrics`
- `list_models(name_filter)` - Returns pandas DataFrame of registered models

**Helper Methods (Critical to Preserve):**
- `_convert_to_dspy_examples(df)` - Converts pandas to DSPy Examples
- `_create_classification_module()` - Creates ChainOfThought module
- `_create_optimizer()` - Factory for BootstrapFewShot or MIPROv2
- `_classification_metric(example, prediction, trace)` - Returns 1.0 or 0.0 accuracy
- `_evaluate_on_dataset(module, examples, split)` - Unified evaluation with artifact logging
- `_log_dataset_info(split, num_examples)` - Logs dataset metadata to MLFlow
- `_log_predictions_artifacts(examples, predictions, split)` - Logs sample predictions (CSV + Parquet)
- `_log_disagreements_artifacts(examples, predictions, split)` - Logs prediction errors
- `_extract_and_log_prompt(compiled_module, train_examples)` - Comprehensive prompt extraction

**MLFlow Integration Patterns:**
- Uses `mlflow.dspy.autolog()` with specific flags (line 65-69)
- Sets experiment with `mlflow.set_experiment()` and tags (line 72-81)
- Logs artifacts in dual format: CSV + Parquet for predictions
- Creates `prompt_details.txt` with detailed prompt information
- Metric naming: `train_accuracy` for training, `accuracy` for validation/test
- Model registration with aliases via MlflowClient

**Key Insight:** The service follows a lazy loading pattern similar to DatasetService, storing cached data in private attributes.

### Current Configuration Models (models/model_development.py)

**Location:** `src/symptom_diagnosis_explorer/models/model_development.py` (269 lines)

**Key Classes:**

1. **OptimizerType** (Enum):
   - `BOOTSTRAP_FEW_SHOT = "bootstrap"`
   - `MIPRO_V2 = "mipro"`

2. **OptimizerConfig** (BaseModel):
   - Framework: Pydantic v2
   - Fields: optimizer_type, num_threads
   - Bootstrap-specific: bootstrap_max_bootstrapped_demos, bootstrap_max_labeled_demos
   - MIPROv2-specific: mipro_auto, mipro_minibatch_size, mipro_minibatch_full_eval_steps, mipro_program_aware_proposer, etc.
   - All fields have detailed descriptions and validation (Field(..., description="..."))

3. **LMConfig** (BaseModel):
   - Fields: model, prompt_model (optional), temperature, max_tokens
   - Default model: "ollama/qwen3:1.7b"
   - Temperature range: 0.0-2.0 (with ge/le validators)

4. **ClassificationConfig** (BaseModel):
   - Combines: lm_config, optimizer_config
   - MLFlow settings: mlflow_experiment_name, mlflow_experiment_project, mlflow_tracking_uri, mlflow_run_id
   - Model settings: model_aliases (list), artifact_sample_size
   - Dataset: dataset_identifier

5. **Response Models:**
   - `TuneMetrics`: train_accuracy, validation_accuracy, num_train_examples, num_val_examples
   - `EvaluateMetrics`: accuracy, num_examples, run_id
   - `ModelInfo`: name, version, run_id, metrics (dict)

**Pattern:** All use Pydantic v2 with Field(..., description="...") for rich metadata.

### Current Test Patterns (tests/integration/services/test_model_development.py)

**Location:** `src/symptom_diagnosis_explorer/tests/integration/services/test_model_development.py` (371 lines)

**Key Patterns:**

1. **Fixtures:**
   - `model_development_config` - Creates ClassificationConfig for testing
   - `model_development_service` - Instantiates service
   - `mlflow_test_dir` - Temporary directory for MLFlow (from conftest.py)
   - `test_model_name` - Ollama model name (from conftest.py)

2. **Test Structure:**
   - Uses pytest markers: `@pytest.mark.integration`, `@pytest.mark.llm`, `@pytest.mark.ollama`, `@pytest.mark.slow`
   - Module-level check for required Ollama models (lines 33-45)
   - Tests use small datasets (5 train, 3 validation) for speed

3. **MLFlow Validation Tests:**
   - `test_mlflow_logging_behavior` - Validates:
     - Experiment tags (system, project)
     - Metric naming (train_accuracy vs accuracy)
     - Dual format artifacts (CSV + Parquet)
     - prompt_details.txt artifact
     - Content of prompt_details.txt (sections: MODULE STRUCTURE, SIGNATURE OVERVIEW, FEW-SHOT EXAMPLES, ACTUAL PROMPT EXAMPLE)

4. **Integration Test Examples:**
   - `test_tune_with_small_dataset` - Verifies full tuning pipeline
   - `test_tune_with_mipro_v2_optimizer` - Tests MIPRO optimizer
   - `test_evaluate_model` - Tests evaluation on test split
   - `test_list_models` - Verifies model registry listing
   - `test_list_models_with_filter` - Tests name filtering

**Key Insight:** Tests are comprehensive and validate both functionality and MLFlow artifact structure.

### Current CLI/Command Structure

**Commands Layer:**
- `commands/classify/tune.py` (171 lines)
  - `TuneRequest` (Pydantic model with CLI parameters)
  - `TuneResponse` (returns metrics, model_info, run_id)
  - `TuneCommand` class with `__init__(request)` and `execute()` methods
  - Pattern: Request → Config → Service → Response

- `commands/classify/evaluate.py` (108 lines)
  - Similar pattern: `EvaluateRequest` → `EvaluateCommand` → `EvaluateResponse`

- `commands/classify/list_models.py` (60 lines)
  - Simpler pattern, no complex config needed

**CLI Layer (cli.py):**
- Uses Typer for CLI framework
- Rich Console for formatted output (tables)
- Command signature: `@classify_app.command("tune")` with `def classify_tune(...)`
- Extensive Annotated parameters with typer.Option
- Constructs full experiment name: `/symptom-diagnosis-explorer/{project}/{experiment_name}`
- Error handling with try/except and `typer.Exit(1)`

**Pattern:** CLI args → Request model → Command → Service → Response model → Rich output

### Current Dependencies (pyproject.toml)

**Relevant ML Libraries:**
- `dspy>=3.0.3` ✅ (Currently installed)
- `langchain>=1.0.5` ✅ (Already in project!)
- `langchain-community>=0.4.1` ✅
- `langchain-ollama>=1.0.0` ✅
- `mlflow>=3.5.0` ✅
- `ollama>=0.6.0` ✅

**Key Insight:** LangChain is already installed! This is excellent - we don't need to add new dependencies for basic LangChain support.

**Missing LangChain Packages (may need to add):**
- `langchain-core` (usually included with langchain)
- `langchain-openai` (if using OpenAI models)

### Dataset Service Pattern (services/dataset.py)

**Pattern to Follow:**
- Lazy loading with private attributes: `_train_df`, `_validation_df`, `_test_df`
- RuntimeError if not loaded: `if self._dataset is None: raise RuntimeError("Dataset not loaded. Call load() first.")`
- Cached splits with deterministic random seed (line 103-107)
- Pandera validation on all DataFrames
- Helper method `_prepare_and_validate_dataframe(df)` for DRY

**Key Takeaway:** New framework services should follow this lazy loading + caching pattern.

---

## Phase 2: External Research

### Topic 1: LangChain LCEL and Structured Output

#### LCEL Chain Construction (2025)

**Basic Syntax:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Modern LCEL syntax (replaces LLMChain)
chain = prompt | llm | parser
```

**Key Features:**
- Pipe operator `|` for composition
- First-class streaming support
- Optimized parallel execution
- Automatic logging to LangSmith
- Replaces traditional `LLMChain` approach

#### Structured Output with Pydantic

**Modern Approach (2025):**
```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class DiagnosisOutput(BaseModel):
    """Classification output for symptom diagnosis."""
    diagnosis: str = Field(description="The predicted diagnosis")
    reasoning: str = Field(description="Explanation of the diagnosis")

model = ChatOpenAI(model="gpt-4")
model_with_structure = model.with_structured_output(DiagnosisOutput)

# LCEL chain with structured output
chain = prompt | model_with_structure
response = chain.invoke({"symptoms": "headache and fever"})
# Returns: DiagnosisOutput(diagnosis="...", reasoning="...")
```

**Key Methods:**
1. **Pydantic Models** (recommended):
   - Use `model.with_structured_output(PydanticModel)`
   - Provides validation and type safety
   - Output is Pydantic model instance

2. **TypedDict** (alternative):
   - Use `model.with_structured_output(TypedDictClass)`
   - No runtime validation
   - Output is plain dictionary

3. **JSON Schema** (maximum control):
   - Use `model.with_structured_output(json_schema, method="json_schema")`
   - Output is dictionary

**Best Practice:** Use Pydantic models for rich validation and nested structures.

#### Few-Shot Prompting in LangChain

**Pattern (from Context7 docs):**
```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Define examples
examples = [
    {"symptoms": "fever, cough, fatigue", "diagnosis": "Common Cold"},
    {"symptoms": "chest pain, shortness of breath", "diagnosis": "Heart Attack"},
]

# Create few-shot template
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{symptoms}"),
    ("ai", "{diagnosis}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Combine with main prompt
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical diagnosis assistant."),
    few_shot_prompt,
    ("human", "{symptoms}"),
])
```

**Note:** LangChain's few-shot prompting is more manual than DSPy's automatic optimization. We need to manually select and format examples.

#### Chain Serialization

**Finding (from MLFlow research):**
- LangChain Runnables can be serialized recursively
- Configurations stored in YAML format
- MLFlow can serialize LCEL chains via `mlflow.langchain.log_model()`
- Limitations: Retrievers are NOT supported by MLFlow autologging

**Pattern:**
```python
# LangChain chains serialize automatically with MLFlow
with mlflow.start_run():
    mlflow.langchain.log_model(
        lc_model=chain,
        name="chain",
        input_example={"symptoms": "fever"}
    )
```

### Topic 2: MLFlow Integration for LangChain

#### Logging LCEL Chains

**Modern Approach (MLFlow 2.14.0+):**
```python
import mlflow

# Enable autologging (logs traces by default)
mlflow.langchain.autolog()

# Optional: Enable model + artifact logging
# mlflow.langchain.autolog(log_models=True, log_input_examples=True)

# Build LCEL chain
chain = prompt | llm | StrOutputParser()

# Invoke chain (automatically logged)
response = chain.invoke({"question": "What is MLflow?"})
```

**What Gets Logged:**
1. **Traces** (default): Full execution trace of chain
2. **Models** (if enabled): Chain structure serialized recursively
3. **Artifacts** (if enabled):
   - Chain configuration (YAML)
   - Input examples
   - Model signatures

**Key Insight:** MLFlow autologging for LangChain was "largely renewed in MLflow 2.14.0" - very recent updates!

#### Artifact Logging Strategy

**For Non-Trainable Models (LangChain):**

**Approach: Log Configuration as Artifact (Don't Use log_model)**
```python
import mlflow
import yaml

with mlflow.start_run():
    # Log prompt configuration as artifact
    prompt_config = {
        "template": prompt.template,
        "examples": examples,
        "model_config": {"temperature": 0.7, "model": "gpt-4"}
    }

    # Log as YAML
    with open("prompt_config.yaml", "w") as f:
        yaml.dump(prompt_config, f)
    mlflow.log_artifact("prompt_config.yaml")

    # Register model for versioning (but don't load from MLFlow)
    # The chain will be recreated from hardcoded templates, not loaded
```

**Important Note:**
- `mlflow.langchain.log_model()` does **NOT** work reliably for loading chains
- Instead: Store prompts in code (`services/ml_models/prompts/langchain.py`)
- Log configuration as YAML artifact for tracking purposes
- Recreate chain from hardcoded templates during evaluation
- This differs from DSPy, which can load compiled models from MLFlow

#### MLFlow Experiment Structure

**Recommended Tagging for Multi-Framework:**
```python
mlflow.set_experiment(experiment_name)
mlflow.set_experiment_tags({
    "system": "symptom-diagnosis-explorer",
    "project": config.mlflow_experiment_project,
    "framework": "langchain",  # NEW TAG
    "trainable": "false"  # NEW TAG
})

# Also log as run parameters
with mlflow.start_run():
    mlflow.log_params({
        "framework": "langchain",
        "requires_training": False,
        "lm_model": config.lm_config.model,
        "temperature": config.lm_config.temperature,
    })
```

**Query Pattern:**
```python
# Search for LangChain runs
mlflow.search_runs(
    experiment_ids=[exp_id],
    filter_string="params.framework = 'langchain'"
)

# Search for non-trainable runs
mlflow.search_runs(
    filter_string="params.requires_training = 'false'"
)
```

#### Token Usage Tracking

**Important Finding (from MLFlow docs):**
```python
# Token usage is automatically tracked in spans
trace = mlflow.get_trace(trace_id=last_trace_id)

# Total usage
total_usage = trace.info.token_usage
# {
#   "input_tokens": 123,
#   "output_tokens": 456,
#   "total_tokens": 579
# }

# Per-LLM-call usage
for span in trace.data.spans:
    if usage := span.get_attribute("mlflow.chat.tokenUsage"):
        print(f"{span.name}: {usage['total_tokens']} tokens")
```

**Application:** We should expose token usage in our `EvaluateMetrics` and `TuneMetrics` responses.

### Topic 3: Pydantic Discriminated Unions

#### Basic Discriminated Union Pattern

**Official Pydantic v2 Syntax (2025):**
```python
from pydantic import BaseModel, Field
from typing import Literal, Union
from typing import Annotated

# Base class with discriminator field
class BaseFrameworkConfig(BaseModel):
    framework: str  # Discriminator field

# Subclasses with Literal discriminator values
class DSPyConfig(BaseFrameworkConfig):
    framework: Literal["dspy"] = "dspy"
    optimizer_config: OptimizerConfig
    # DSPy-specific fields

class LangChainConfig(BaseFrameworkConfig):
    framework: Literal["langchain"] = "langchain"
    few_shot_examples: list[dict] | None = None
    # LangChain-specific fields

# Use Annotated with discriminator
FrameworkConfig = Annotated[
    Union[DSPyConfig, LangChainConfig],
    Field(discriminator="framework")
]

# Usage
class ClassificationConfig(BaseModel):
    lm_config: LMConfig
    framework_config: FrameworkConfig  # Will validate correctly!
    mlflow_experiment_name: str
```

**How It Works:**
- Pydantic validates based on the `framework` field value
- If `framework="dspy"`, validates against `DSPyConfig`
- If `framework="langchain"`, validates against `LangChainConfig`
- Much faster than trying each union member sequentially

#### Advanced: Callable Discriminators

**For Complex Cases:**
```python
from pydantic import Discriminator

def get_framework_type(v: Any) -> str:
    # Custom logic to determine discriminator
    if isinstance(v, dict):
        return v.get("framework", "unknown")
    return "unknown"

FrameworkConfig = Annotated[
    Union[DSPyConfig, LangChainConfig],
    Discriminator(get_framework_type)
]
```

#### Fallback Pattern (2025)

**For Unknown Frameworks:**
```python
class GenericFrameworkConfig(BaseFrameworkConfig):
    """Fallback for unknown frameworks."""
    framework: str
    config: dict  # Generic config storage

# Nested union with fallback
FrameworkConfig = Annotated[
    Union[
        # Try specific types first
        Annotated[
            Union[DSPyConfig, LangChainConfig],
            Field(discriminator="framework")
        ],
        # Fallback to generic
        GenericFrameworkConfig
    ],
    Field(union_mode="left_to_right")
]
```

**Key Insight:** This pattern from 2025 allows graceful handling of future frameworks without breaking validation.

#### Computed Properties

**Pattern:**
```python
from pydantic import computed_field

class DSPyConfig(BaseFrameworkConfig):
    framework: Literal["dspy"] = "dspy"
    optimizer_config: OptimizerConfig

    @computed_field
    @property
    def requires_training(self) -> bool:
        """DSPy always requires training."""
        return True

class LangChainConfig(BaseFrameworkConfig):
    framework: Literal["langchain"] = "langchain"

    @computed_field
    @property
    def requires_training(self) -> bool:
        """LangChain prompt engineering doesn't require training."""
        return False
```

**Application:** Use `@computed_field` instead of regular `@property` for serialization support.

### Topic 4: Registry Pattern

#### Decorator-Based Registration

**Type-Safe Pattern (2025):**
```python
from typing import Type, Dict, Callable
from pydantic import BaseModel

# Type alias for clarity
ServiceFactory = Callable[[BaseModel], "BaseModelService"]

class ModelServiceRegistry:
    """Registry for model development services."""

    _registry: Dict[str, Type["BaseModelService"]] = {}

    @classmethod
    def register(cls, framework: str) -> Callable[[Type["BaseModelService"]], Type["BaseModelService"]]:
        """Decorator to register a service class.

        Usage:
            @ModelServiceRegistry.register("dspy")
            class DSPyModelService(BaseModelService):
                ...
        """
        def decorator(service_class: Type["BaseModelService"]) -> Type["BaseModelService"]:
            cls._registry[framework] = service_class
            return service_class
        return decorator

    @classmethod
    def create_service(cls, framework: str, config: BaseModel) -> "BaseModelService":
        """Factory method to create service instance.

        Args:
            framework: Framework identifier (e.g., "dspy", "langchain")
            config: Configuration object (type varies by framework)

        Returns:
            Instantiated service

        Raises:
            ValueError: If framework not registered
        """
        if framework not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown framework: {framework}. "
                f"Available frameworks: {available}"
            )

        service_class = cls._registry[framework]
        return service_class(config)

    @classmethod
    def list_frameworks(cls) -> list[str]:
        """List all registered frameworks."""
        return list(cls._registry.keys())
```

**Usage:**
```python
# Register services
@ModelServiceRegistry.register("dspy")
class DSPyModelService(BaseModelService):
    def __init__(self, config: DSPyConfig):
        ...

@ModelServiceRegistry.register("langchain")
class LangChainModelService(BaseModelService):
    def __init__(self, config: LangChainConfig):
        ...

# Factory usage
service = ModelServiceRegistry.create_service(
    framework="dspy",
    config=dspy_config
)
```

**Type Safety Challenges (from GitHub discussion):**
- Can't fully preserve specific type information through decorator transformation
- `ParamSpec` and `Concatenate` (PEP 612, Python 3.10+) help but don't solve all cases
- Recommendation: Use `Type["BaseModelService"]` and accept some type checker limitations

**Best Practice:** Use registration for discoverability, but validate config types at runtime.

#### Error Handling

**Pattern:**
```python
class FrameworkNotRegisteredError(ValueError):
    """Raised when framework is not in registry."""
    pass

@classmethod
def create_service(cls, framework: str, config: BaseModel) -> "BaseModelService":
    if framework not in cls._registry:
        raise FrameworkNotRegisteredError(
            f"Framework '{framework}' not registered. "
            f"Available: {', '.join(cls._registry.keys())}"
        )
    # ...
```

### Topic 5: Abstract Base Classes

#### Python ABC Pattern

**Pattern for Our Use Case:**
```python
from abc import ABC, abstractmethod
from typing import Protocol

class BaseModelService(ABC):
    """Abstract base class for model development services."""

    def __init__(self, config: BaseModel):
        """Initialize service with config.

        Args:
            config: Framework-specific configuration
        """
        self.config = config
        self.dataset_service = DatasetService()

    @abstractmethod
    def tune(
        self,
        train_size: int | None = None,
        val_size: int | None = None,
        model_name: str = "symptom-classifier",
    ) -> tuple[TuneMetrics, ModelInfo]:
        """Tune/optimize classification model.

        Args:
            train_size: Optional limit on training examples
            val_size: Optional limit on validation examples
            model_name: Name for model registry

        Returns:
            Tuple of (metrics, model_info)
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model_name: str,
        model_version: str | None = None,
        split: str = "test",
        eval_size: int | None = None,
    ) -> EvaluateMetrics:
        """Evaluate a saved model.

        Args:
            model_name: Name in registry
            model_version: Specific version or None for latest
            split: Dataset split to evaluate on
            eval_size: Optional limit on examples

        Returns:
            Evaluation metrics
        """
        pass

    @abstractmethod
    def list_models(self, name_filter: str | None = None) -> pd.DataFrame:
        """List registered models from MLFlow.

        Args:
            name_filter: Optional substring filter

        Returns:
            DataFrame with model information
        """
        pass

    @property
    @abstractmethod
    def requires_training(self) -> bool:
        """Whether this framework requires training/optimization.

        Returns:
            True for DSPy (requires compile), False for LangChain (prompt-based)
        """
        pass

    # Concrete helper methods (shared across frameworks)
    def _log_dataset_info(self, split: str, num_examples: int) -> None:
        """Log dataset information to MLFlow."""
        mlflow.log_param(f"dataset_{split}_source", self.config.dataset_identifier)
        mlflow.log_param(f"dataset_{split}_size", num_examples)
```

**Key Points:**
- Use `@abstractmethod` for required methods
- Use `@property` + `@abstractmethod` for required properties
- Concrete methods can access abstract properties
- Subclasses MUST implement all abstract methods/properties

---

## Phase 3: Synthesis and Recommendations

### Recommended Architecture

#### 1. Configuration Models Structure

**File:** `models/model_development.py`

```python
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Union, Annotated
from enum import Enum

# Existing (keep as-is)
class LMConfig(BaseModel):
    model: str = Field(default="ollama/qwen3:1.7b")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1_024, gt=0)
    # Note: prompt_model used only by DSPy MIPRO v2

# NEW: Base class for framework configs
class BaseFrameworkConfig(BaseModel):
    """Base class for framework-specific configuration."""
    framework: str

    @computed_field
    @property
    def requires_training(self) -> bool:
        """Whether this framework requires training/optimization."""
        raise NotImplementedError

# Existing (keep as-is, but make it extend base)
class OptimizerType(str, Enum):
    BOOTSTRAP_FEW_SHOT = "bootstrap"
    MIPRO_V2 = "mipro"

class OptimizerConfig(BaseModel):
    # ... existing fields ...
    pass

class DSPyFrameworkConfig(BaseFrameworkConfig):
    """DSPy-specific configuration."""
    framework: Literal["dspy"] = "dspy"
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)

    @computed_field
    @property
    def requires_training(self) -> bool:
        return True  # DSPy always requires compilation

# NEW: LangChain configuration
class LangChainFrameworkConfig(BaseFrameworkConfig):
    """LangChain LCEL-specific configuration."""
    framework: Literal["langchain"] = "langchain"
    few_shot_examples: list[dict] | None = Field(
        default=None,
        description="Optional few-shot examples for prompt template"
    )
    system_prompt: str = Field(
        default="You are a medical diagnosis assistant.",
        description="System prompt for the chat model"
    )

    @computed_field
    @property
    def requires_training(self) -> bool:
        return False  # LangChain uses prompt engineering

# Discriminated union
FrameworkConfig = Annotated[
    Union[DSPyFrameworkConfig, LangChainFrameworkConfig],
    Field(discriminator="framework")
]

# Updated ClassificationConfig
class ClassificationConfig(BaseModel):
    """Top-level configuration supporting multiple frameworks."""
    lm_config: LMConfig = Field(default_factory=LMConfig)
    framework_config: FrameworkConfig  # Discriminated union!
    mlflow_experiment_name: str
    mlflow_experiment_project: str
    mlflow_tracking_uri: str = Field(default="http://localhost:5001")
    mlflow_run_id: str | None = Field(default=None)
    model_aliases: list[str] = Field(default_factory=list)
    artifact_sample_size: int = Field(default=10, gt=0)
    dataset_identifier: str = Field(default="gretelai/symptom_to_diagnosis")
```

**Key Benefits:**
- Type-safe framework selection via discriminator
- Shared fields in `ClassificationConfig`
- Framework-specific fields in subclasses
- Computed property for `requires_training`

#### 2. Service Layer Structure

**File:** `services/base_model_service.py` (NEW)

```python
from abc import ABC, abstractmethod
import pandas as pd
from pydantic import BaseModel

class BaseModelService(ABC):
    """Abstract base class for model development services.

    All framework-specific services must extend this class and
    implement the required abstract methods.
    """

    def __init__(self, config: BaseModel):
        self.config = config
        self.dataset_service = DatasetService()

    @abstractmethod
    def tune(
        self,
        train_size: int | None = None,
        val_size: int | None = None,
        model_name: str = "symptom-classifier",
    ) -> tuple[TuneMetrics, ModelInfo]:
        """Tune/optimize classification model."""
        pass

    @abstractmethod
    def evaluate(
        self,
        model_name: str,
        model_version: str | None = None,
        split: str = "test",
        eval_size: int | None = None,
    ) -> EvaluateMetrics:
        """Evaluate a saved model."""
        pass

    @abstractmethod
    def list_models(self, name_filter: str | None = None) -> pd.DataFrame:
        """List registered models from MLFlow registry."""
        pass

    @property
    @abstractmethod
    def requires_training(self) -> bool:
        """Whether this framework requires training/optimization."""
        pass

    # Shared helper methods (concrete implementations)
    def _log_dataset_info(self, split: str, num_examples: int) -> None:
        """Log dataset information to MLFlow."""
        import mlflow
        mlflow.log_param(f"dataset_{split}_source", self.config.dataset_identifier)
        mlflow.log_param(f"dataset_{split}_size", num_examples)
```

**File:** `services/dspy_model_service.py` (REFACTOR existing)

```python
from services.base_model_service import BaseModelService
from services.registry import ModelServiceRegistry

@ModelServiceRegistry.register("dspy")
class DSPyModelService(BaseModelService):
    """DSPy-based model development service."""

    def __init__(self, config: DSPyFrameworkConfig):
        # Wrap in ClassificationConfig for compatibility
        self.framework_config = config
        super().__init__(config)

        # Existing DSPy setup
        lm = dspy.LM(...)
        dspy.configure(lm=lm)
        mlflow.dspy.autolog(...)
        # etc.

    @property
    def requires_training(self) -> bool:
        return True

    # Move all existing methods from ModelDevelopmentService here
    def tune(self, ...) -> tuple[TuneMetrics, ModelInfo]:
        # Existing implementation
        pass

    def evaluate(self, ...) -> EvaluateMetrics:
        # Existing implementation
        pass

    def list_models(self, ...) -> pd.DataFrame:
        # Existing implementation
        pass
```

**File:** `services/langchain_model_service.py` (NEW)

```python
from services.base_model_service import BaseModelService
from services.registry import ModelServiceRegistry
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

@ModelServiceRegistry.register("langchain")
class LangChainModelService(BaseModelService):
    """LangChain LCEL-based model development service."""

    def __init__(self, config: LangChainFrameworkConfig):
        self.framework_config = config
        super().__init__(config)

        # Setup MLFlow autologging for LangChain
        import mlflow
        mlflow.langchain.autolog()
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)
        mlflow.set_experiment_tags({
            "system": "symptom-diagnosis-explorer",
            "project": config.mlflow_experiment_project,
            "framework": "langchain",
            "trainable": "false",
        })

    @property
    def requires_training(self) -> bool:
        return False

    def _create_classification_chain(self):
        """Create LCEL chain with structured output."""

        # Define output schema
        class DiagnosisOutput(BaseModel):
            diagnosis: str = Field(description="The predicted diagnosis")

        # Create prompt template
        system_prompt = self.framework_config.system_prompt

        if self.framework_config.few_shot_examples:
            # Build few-shot prompt
            # (Implementation details)
            pass
        else:
            # Simple zero-shot prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Patient symptoms: {symptoms}\\nDiagnosis:"),
            ])

        # Create chain with structured output
        llm = ChatOpenAI(
            model=self.config.lm_config.model,
            temperature=self.config.lm_config.temperature,
            max_tokens=self.config.lm_config.max_tokens,
        )
        model_with_structure = llm.with_structured_output(DiagnosisOutput)
        chain = prompt | model_with_structure

        return chain

    def tune(
        self,
        train_size: int | None = None,
        val_size: int | None = None,
        model_name: str = "symptom-classifier",
    ) -> tuple[TuneMetrics, ModelInfo]:
        """For LangChain, 'tuning' means prompt engineering + evaluation.

        Since LangChain doesn't require training, this method:
        1. Builds the chain
        2. Evaluates on train and validation sets
        3. Logs the chain to MLFlow
        4. Returns metrics
        """

        # Load datasets
        self.dataset_service.load()
        train_df, val_df = self.dataset_service.get_train_validation_split()
        if train_size:
            train_df = train_df.head(train_size)
        if val_size:
            val_df = val_df.head(val_size)

        # Create chain
        chain = self._create_classification_chain()

        # Evaluate on both splits
        with mlflow.start_run(run_id=self.config.mlflow_run_id) as run:
            # Log parameters
            mlflow.log_params({
                "framework": "langchain",
                "requires_training": False,
                "lm_model": self.config.lm_config.model,
                "train_size": len(train_df),
                "val_size": len(val_df),
            })

            # Evaluate
            train_accuracy = self._evaluate_on_dataset(chain, train_df, "train")
            val_accuracy = self._evaluate_on_dataset(chain, val_df, "validation")

            # Log prompt/config as artifact (don't use log_model - it doesn't work)
            prompt_config = {
                "template_name": self.framework_config.prompt_template_name,
                "few_shot_examples": few_shot_examples,
                "model": self.config.lm_config.model,
                "temperature": self.config.lm_config.temperature,
            }
            with open("prompt_config.yaml", "w") as f:
                yaml.dump(prompt_config, f)
            mlflow.log_artifact("prompt_config.yaml")

            # Register model (for versioning, but won't load from MLFlow)
            model_uri = f"runs:/{run.info.run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)

            # Set aliases
            client = mlflow.tracking.MlflowClient()
            for alias in self.config.model_aliases:
                client.set_registered_model_alias(
                    model_name, alias, model_version.version
                )

            # Build response
            metrics = TuneMetrics(
                train_accuracy=train_accuracy,
                validation_accuracy=val_accuracy,
                num_train_examples=len(train_df),
                num_val_examples=len(val_df),
            )

            model_info = ModelInfo(
                name=model_name,
                version=str(model_version.version),
                run_id=run.info.run_id,
                metrics={
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": val_accuracy,
                },
            )

            return metrics, model_info

    def _evaluate_on_dataset(
        self,
        chain,
        df: pd.DataFrame,
        split: str,
    ) -> float:
        """Evaluate chain on dataset and log artifacts."""
        predictions = []
        actuals = []

        for _, row in df.iterrows():
            result = chain.invoke({"symptoms": row["symptoms"]})
            predictions.append(result.diagnosis)
            actuals.append(row["diagnosis"])

        # Calculate accuracy
        correct = sum(
            pred.strip().lower() == actual.strip().lower()
            for pred, actual in zip(predictions, actuals)
        )
        accuracy = correct / len(df)

        # Log metrics
        metric_name = "train_accuracy" if split == "train" else "accuracy"
        mlflow.log_metric(metric_name, accuracy)

        # Log dataset info and artifacts (similar to DSPy)
        self._log_dataset_info(split, len(df))
        # ... additional artifact logging ...

        return accuracy

    def evaluate(
        self,
        model_name: str,
        model_version: str | None = None,
        split: str = "test",
        eval_size: int | None = None,
    ) -> EvaluateMetrics:
        """Evaluate a saved LangChain model."""

        # Load dataset split
        self.dataset_service.load()
        if split == "train":
            df = self.dataset_service.get_train_dataframe()
        elif split == "validation":
            df = self.dataset_service.get_validation_dataframe()
        elif split == "test":
            df = self.dataset_service.get_test_dataframe()
        else:
            raise ValueError(f"Invalid split: {split}")

        if eval_size:
            df = df.head(eval_size)

        # Load model
        if model_version:
            model_uri = f"models:/{model_name}/{model_version}"
        else:
            model_uri = f"models:/{model_name}/latest"

        chain = mlflow.langchain.load_model(model_uri)

        # Evaluate
        with mlflow.start_run(run_id=self.config.mlflow_run_id) as run:
            mlflow.log_params({
                "model_name": model_name,
                "model_version": model_version or "latest",
                "split": split,
            })

            accuracy = self._evaluate_on_dataset(chain, df, split)

            return EvaluateMetrics(
                accuracy=accuracy,
                num_examples=len(df),
                run_id=run.info.run_id,
            )

    def list_models(self, name_filter: str | None = None) -> pd.DataFrame:
        """List LangChain models from MLFlow registry."""
        # Same implementation as DSPy service
        # (Can be moved to base class if identical)
        pass
```

#### 3. Registry Pattern Implementation

**File:** `services/registry.py` (NEW)

```python
from typing import Type, Dict, Callable
from pydantic import BaseModel
from services.base_model_service import BaseModelService

class ModelServiceRegistry:
    """Registry for model development services.

    Usage:
        # Register a service
        @ModelServiceRegistry.register("dspy")
        class DSPyModelService(BaseModelService):
            ...

        # Create service instance
        service = ModelServiceRegistry.create_service("dspy", config)
    """

    _registry: Dict[str, Type[BaseModelService]] = {}

    @classmethod
    def register(
        cls, framework: str
    ) -> Callable[[Type[BaseModelService]], Type[BaseModelService]]:
        """Decorator to register a service class."""
        def decorator(service_class: Type[BaseModelService]) -> Type[BaseModelService]:
            cls._registry[framework] = service_class
            return service_class
        return decorator

    @classmethod
    def create_service(
        cls, framework: str, config: BaseModel
    ) -> BaseModelService:
        """Factory method to create service instance."""
        if framework not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown framework: {framework}. "
                f"Available frameworks: {available}"
            )

        service_class = cls._registry[framework]
        return service_class(config)

    @classmethod
    def list_frameworks(cls) -> list[str]:
        """List all registered frameworks."""
        return list(cls._registry.keys())
```

#### 4. Command Layer Updates

**File:** `commands/classify/tune.py`

```python
# Add framework parameter to TuneRequest
class TuneRequest(BaseModel):
    framework: Literal["dspy", "langchain"] = Field(
        default="dspy",
        description="Framework to use for classification"
    )
    # ... existing fields ...

    # LangChain-specific (optional)
    few_shot_examples: list[dict] | None = Field(
        default=None,
        description="Few-shot examples for LangChain (ignored for DSPy)"
    )
    system_prompt: str = Field(
        default="You are a medical diagnosis assistant.",
        description="System prompt for LangChain (ignored for DSPy)"
    )

class TuneCommand:
    def __init__(self, request: TuneRequest) -> None:
        # Build framework-specific config
        if request.framework == "dspy":
            framework_config = DSPyFrameworkConfig(
                optimizer_config=OptimizerConfig(...)
            )
        elif request.framework == "langchain":
            framework_config = LangChainFrameworkConfig(
                few_shot_examples=request.few_shot_examples,
                system_prompt=request.system_prompt,
            )
        else:
            raise ValueError(f"Unknown framework: {request.framework}")

        # Build top-level config
        config = ClassificationConfig(
            lm_config=LMConfig(model=request.lm_model),
            framework_config=framework_config,
            mlflow_experiment_name=request.experiment_name,
            mlflow_experiment_project=request.experiment_project,
            mlflow_tracking_uri=request.mlflow_tracking_uri,
        )

        # Create service via registry
        from services.registry import ModelServiceRegistry
        self.service = ModelServiceRegistry.create_service(
            framework=request.framework,
            config=config,
        )
        self.request = request

    def execute(self) -> TuneResponse:
        # Same as before
        metrics, model_info = self.service.tune(...)
        return TuneResponse(...)
```

#### 5. CLI Updates

**File:** `cli.py`

```python
@classify_app.command("tune")
def classify_tune(
    framework: Annotated[
        Literal["dspy", "langchain"],
        typer.Option(
            help="Framework to use (dspy or langchain)",
        ),
    ] = "dspy",
    # ... existing parameters ...

    # LangChain-specific options
    few_shot_examples_file: Annotated[
        Optional[str],
        typer.Option(
            help="Path to JSON file with few-shot examples (LangChain only)",
        ),
    ] = None,
    system_prompt: Annotated[
        str,
        typer.Option(
            help="System prompt for LangChain (ignored for DSPy)",
        ),
    ] = "You are a medical diagnosis assistant.",
) -> None:
    """Tune classification model using DSPy or LangChain."""

    # Load few-shot examples if provided
    few_shot_examples = None
    if few_shot_examples_file:
        import json
        with open(few_shot_examples_file) as f:
            few_shot_examples = json.load(f)

    # Create request
    request = TuneRequest(
        framework=framework,
        few_shot_examples=few_shot_examples,
        system_prompt=system_prompt,
        # ... other parameters ...
    )

    # Execute command (rest unchanged)
    command = TuneCommand(request)
    response = command.execute()
    # ... display results ...
```

### Implementation Checklist

- [ ] Create `services/base_model_service.py` with ABC
- [ ] Create `services/registry.py` with registration pattern
- [ ] Refactor `services/model_development.py` → `services/dspy_model_service.py`
  - [ ] Extend `BaseModelService`
  - [ ] Add `@ModelServiceRegistry.register("dspy")` decorator
  - [ ] Keep all existing functionality
- [ ] Create `services/langchain_model_service.py`
  - [ ] Extend `BaseModelService`
  - [ ] Add `@ModelServiceRegistry.register("langchain")` decorator
  - [ ] Implement `tune()`, `evaluate()`, `list_models()`
  - [ ] Use `mlflow.langchain.autolog()`
- [ ] Update `models/model_development.py`
  - [ ] Add `BaseFrameworkConfig`
  - [ ] Add `DSPyFrameworkConfig`
  - [ ] Add `LangChainFrameworkConfig`
  - [ ] Create discriminated union `FrameworkConfig`
  - [ ] Update `ClassificationConfig` to use `FrameworkConfig`
- [ ] Update `commands/classify/tune.py`
  - [ ] Add `framework` field to `TuneRequest`
  - [ ] Add LangChain-specific fields
  - [ ] Update `TuneCommand` to use registry
- [ ] Update `commands/classify/evaluate.py`
  - [ ] Add `framework` field to `EvaluateRequest`
  - [ ] Update `EvaluateCommand` to use registry
- [ ] Update `cli.py`
  - [ ] Add `--framework` option
  - [ ] Add `--few-shot-examples-file` option
  - [ ] Add `--system-prompt` option
- [ ] Create tests for LangChain service
  - [ ] `tests/integration/services/test_langchain_model_service.py`
  - [ ] Mirror structure of `test_model_development.py`
  - [ ] Test `tune()` with small dataset
  - [ ] Test `evaluate()` on test split
  - [ ] Test MLFlow artifact logging
- [ ] Update documentation
  - [ ] README with examples for both frameworks
  - [ ] Migration guide for existing DSPy users

### Migration Strategy

**Phase 1: Create abstractions (no breaking changes)**
1. Create `base_model_service.py` and `registry.py`
2. Keep existing `ModelDevelopmentService` as-is
3. Add discriminated unions to models (backward compatible)

**Phase 2: Refactor DSPy (minimal breaking changes)**
1. Create `DSPyModelService` extending base
2. Update imports but keep `ModelDevelopmentService` as alias
3. Deprecation warning for old import path

**Phase 3: Add LangChain support**
1. Implement `LangChainModelService`
2. Update commands to support both frameworks
3. Update CLI with new options

**Phase 4: Testing and validation**
1. Comprehensive integration tests
2. Verify MLFlow compatibility
3. Performance testing

### Key Design Decisions

1. **Why discriminated unions?**
   - Type-safe framework selection
   - Automatic validation by Pydantic
   - Clear error messages for configuration issues
   - Future-proof for adding more frameworks

2. **Why registry pattern?**
   - Decoupled framework addition
   - Easy discovery of available frameworks
   - Plugin-like architecture
   - Testability (can mock registry)

3. **Why ABC with concrete helpers?**
   - Enforces consistent interface
   - Allows code sharing (e.g., `_log_dataset_info`)
   - Clear contract for new frameworks
   - Type safety with mypy

4. **Why NOT trainable for LangChain?**
   - LangChain LCEL chains are deterministic given the same prompt
   - No optimization loop (unlike DSPy's compile)
   - "Tuning" is really prompt engineering + evaluation
   - Simplifies mental model: DSPy trains, LangChain evaluates

5. **Why log LangChain chains to MLFlow?**
   - Consistent with DSPy approach
   - Enables model versioning and comparison
   - MLFlow autologging handles serialization
   - Can load and reuse chains

### Potential Gotchas

1. **MLFlow LangChain Limitations:**
   - Retrievers NOT supported by autologging
   - Must manually log if using retrievers
   - Solution: We're not using retrievers, so N/A

2. **Type hints with registry:**
   - Can't perfectly preserve types through decorator
   - Solution: Accept type checker limitations, validate at runtime

3. **LangChain few-shot examples:**
   - Not automatically optimized like DSPy
   - Manual selection required
   - Solution: Provide clear documentation on example selection

4. **Metric consistency:**
   - DSPy and LangChain may produce different accuracies
   - Solution: Clearly document that frameworks are not directly comparable

5. **Token usage tracking:**
   - LangChain autologging provides token usage
   - DSPy does not automatically track tokens
   - Solution: Consider exposing token usage only for LangChain

### Testing Strategy

**Unit Tests:**
- Registry registration and lookup
- Discriminated union validation
- Config model validation
- ABC enforcement

**Integration Tests:**
- Full tune → evaluate → list cycle for LangChain
- MLFlow artifact validation
- Cross-framework model listing
- Error handling (unknown framework, invalid config)

**Comparison Tests:**
- Same dataset, both frameworks
- Verify metrics are reasonable
- Verify MLFlow experiments are tagged correctly

---

## Code Examples for Key Patterns

### Example 1: Using the Registry

```python
from services.registry import ModelServiceRegistry
from models.model_development import DSPyFrameworkConfig, LangChainFrameworkConfig

# DSPy service
dspy_config = DSPyFrameworkConfig(
    optimizer_config=OptimizerConfig(...)
)
dspy_service = ModelServiceRegistry.create_service("dspy", dspy_config)

# LangChain service
langchain_config = LangChainFrameworkConfig(
    few_shot_examples=[...],
    system_prompt="You are a medical assistant.",
)
langchain_service = ModelServiceRegistry.create_service("langchain", langchain_config)

# Both have the same interface
metrics, model_info = dspy_service.tune(train_size=100)
metrics, model_info = langchain_service.tune(train_size=100)
```

### Example 2: Discriminated Union Validation

```python
from models.model_development import ClassificationConfig, DSPyFrameworkConfig

# Valid DSPy config
config = ClassificationConfig(
    framework_config=DSPyFrameworkConfig(framework="dspy"),
    mlflow_experiment_name="/symptom-diagnosis-explorer/test/exp",
    mlflow_experiment_project="test",
)
# ✅ Validates successfully

# Invalid: wrong discriminator
config = ClassificationConfig(
    framework_config={"framework": "unknown", ...},
    ...
)
# ❌ Pydantic ValidationError: "Input should be 'dspy' or 'langchain'"
```

### Example 3: CLI Usage

```bash
# DSPy (existing behavior)
symptom-diagnosis-explorer classify tune \
    --framework dspy \
    --optimizer bootstrap \
    --train-size 100 \
    --val-size 20

# LangChain (new)
symptom-diagnosis-explorer classify tune \
    --framework langchain \
    --train-size 100 \
    --val-size 20 \
    --system-prompt "You are an expert medical diagnosis assistant." \
    --few-shot-examples-file examples.json

# Compare results
symptom-diagnosis-explorer classify list-models
```

---

## Appendix: Research Sources

### Primary Sources

1. **Pydantic Documentation (2025)**
   - https://docs.pydantic.dev/latest/concepts/unions/
   - Discriminated unions with Field(discriminator="...")
   - Callable discriminators for complex cases
   - Fallback patterns with union_mode="left_to_right"

2. **LangChain Documentation (2025)**
   - https://python.langchain.com/docs/concepts/lcel/
   - LCEL syntax: prompt | llm | parser
   - with_structured_output() for Pydantic models
   - Few-shot prompting with FewShotChatMessagePromptTemplate

3. **MLFlow LangChain Integration (v2.14.0+)**
   - https://mlflow.org/docs/latest/genai/flavors/langchain/
   - mlflow.langchain.autolog() for automatic logging
   - log_model() for chain serialization
   - Token usage tracking in traces

4. **GitHub Python Typing Discussions**
   - https://github.com/python/typing/discussions/1565
   - Registry pattern type hints challenges
   - ParamSpec and Concatenate usage

### Key Findings Summary

- **LangChain packages are already installed** in pyproject.toml
- **MLFlow 2.14.0 renewed LangChain autologging** - very recent!
- **Pydantic v2 discriminated unions** are production-ready
- **Registry pattern** is well-established but has type hint limitations
- **DSPy and LangChain are fundamentally different**: DSPy optimizes prompts, LangChain uses fixed prompts
- **Token usage tracking** is automatic with MLFlow LangChain integration

### Recommended Reading

1. **For Pydantic patterns:**
   - https://typethepipe.com/post/pydantic-discriminated-union/
   - https://www.lowlevelmanager.com/2025/05/pydantic-v2-discriminated-unions.html

2. **For LangChain LCEL:**
   - https://www.aurelio.ai/learn/langchain-lcel
   - https://medium.com/@dharamai2024/langchain-chains-expression-language-lcel-with-runnables-build-smart-modular-llm-0d6fb3ee82d8

3. **For MLFlow integration:**
   - https://mlflow.org/docs/latest/genai/flavors/langchain/autologging/
   - https://python.langchain.com/docs/integrations/providers/mlflow_tracking/

---

## Next Steps

1. **Review this document** with stakeholders
2. **Clarify any ambiguous requirements** (e.g., few-shot example selection strategy)
3. **Prioritize implementation phases** (recommend Phase 1-2 first)
4. **Begin implementation** following the checklist above
5. **Iterate based on testing** and real-world usage

---

**Document Status:** ✅ Research Complete
**Last Updated:** 2025-11-09
**Author:** Claude Code
**Version:** 1.0
