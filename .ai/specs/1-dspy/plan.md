# Plan: Integrate DSPy for Simple Classification with MLFlow Tracking

## Related Issue
Issue #1: spike: integrate DSPy for simple classification with MLFlow tracking

## Overview
Implement DSPy-based classification system with MLFlow experiment tracking and model registry integration. Create four CLI commands (`tune`, `evaluate`, `predict`, `list-models`) that follow existing architectural patterns and support Jupyter notebook workflows.

## Architecture Overview

```
CLI Layer (cli.py)
    ↓
Commands Layer (commands/classify/)
    ↓
Services Layer (services/classification.py)
    ↓
Models Layer (models/classification.py)
    ↓
External: DSPy + MLFlow
```

## Implementation Steps

### Step 1: Extend DatasetService
**File:** `src/symptom_diagnosis_explorer/services/dataset.py`

**Changes:**
- Add `get_train_validation_split(train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]`
  - Split existing train data 80/20
  - Use deterministic random seed for reproducibility
  - Cache split results to avoid recomputation
- Add `get_validation_dataframe() -> pd.DataFrame`
  - Return cached validation split
  - Lazy loading pattern (same as existing methods)

**Rationale:** DSPy optimizers need separate train/validation sets. Keep dataset splits in service layer following existing patterns.

---

### Step 2: Create Classification Models
**File:** `src/symptom_diagnosis_explorer/models/classification.py` (new)

**Components:**

**1. DSPy Signature:**
```python
class SymptomDiagnosisSignature(dspy.Signature):
    """Classify symptoms to diagnosis category."""

    symptoms: str = dspy.InputField(desc="Patient symptoms description")
    diagnosis: Literal[<all 22 DiagnosisType values>] = dspy.OutputField(desc="Diagnosis category")
```

**2. Configuration Models:**
```python
class OptimizerType(str, Enum):
    BOOTSTRAP_FEW_SHOT = "bootstrap"
    MIPRO_V2 = "mipro"

class OptimizerConfig(BaseModel):
    optimizer_type: OptimizerType
    num_threads: int = Field(default=4, gt=0)
    max_bootstrapped_demos: int = Field(default=3, ge=0)
    max_labeled_demos: int = Field(default=4, ge=0)
    metric_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

class LMConfig(BaseModel):
    model: str = Field(default="openai/gpt-4o-mini")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=150, gt=0)

class ClassificationConfig(BaseModel):
    lm_config: LMConfig
    optimizer_config: OptimizerConfig
    mlflow_experiment_name: str
    mlflow_tracking_uri: str | None = None
```

**3. Result Models:**
```python
class ClassificationResult(BaseModel):
    diagnosis: str
    confidence: float | None = None

class TuneMetrics(BaseModel):
    train_accuracy: float
    validation_accuracy: float
    num_train_examples: int
    num_val_examples: int

class ModelInfo(BaseModel):
    name: str
    version: str
    run_id: str
    metrics: dict[str, float]
```

**Rationale:** Type-safe configuration and results following existing Pydantic patterns. Mirrors DiagnosisType/SymptomDiagnosisExample structure.

---

### Step 3: Create ClassificationService
**File:** `src/symptom_diagnosis_explorer/services/classification.py` (new)

**Class:** `ClassificationService`

**Dependencies:**
- `DatasetService` for loading datasets
- DSPy for LM configuration and modules
- MLFlow for experiment tracking and model registry

**Key Methods:**

**1. `__init__(config: ClassificationConfig)`**
- Configure DSPy LM: `dspy.configure(lm=dspy.LM(...))`
- Setup MLFlow autologging: `mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)`
- Set tracking URI and experiment

**2. `tune(train_size: int | None, val_size: int | None, model_name: str) -> tuple[TuneMetrics, ModelInfo]`**
- Load train/validation data from DatasetService
- Limit to train_size/val_size if specified
- Convert DataFrames to DSPy Examples
- Create classification module (start with `dspy.ChainOfThought`)
- Initialize optimizer (BootstrapFewShot or MIPROv2)
- Compile/optimize program with `mlflow.start_run()` context
- Evaluate on validation set
- Save to MLFlow model registry with "production" tag
- Return metrics and model info

**3. `evaluate(model_name: str, model_version: str | None, split: str) -> dict[str, float]`**
- Load dataset split (train/validation/test)
- Load model from MLFlow registry by name/version
- Run evaluation with DSPy's `Evaluate` API
- Return accuracy and additional metrics
- Log results to MLFlow

**4. `list_models(name_filter: str | None) -> pd.DataFrame`**
- Query MLFlow model registry
- Return DataFrame with: name, version, stage, creation_time, metrics
- Filter by name pattern if provided

**Helper Methods:**
- `_convert_to_dspy_examples(df: pd.DataFrame) -> list[dspy.Example]`
- `_create_classification_module() -> dspy.Module`
- `_create_optimizer() -> dspy.Teleprompter`
- `_classification_metric(example: dspy.Example, prediction: dspy.Prediction) -> float`

**Rationale:** Encapsulate all DSPy and MLFlow logic in service layer. Follow existing service patterns (lazy loading, caching, typed returns).

---

### Step 4: Create Classify Commands
**Directory:** `src/symptom_diagnosis_explorer/commands/classify/` (new)

**4.1: tune.py**

```python
class TuneRequest(BaseModel):
    optimizer: OptimizerType = Field(default=OptimizerType.BOOTSTRAP_FEW_SHOT)
    train_size: int | None = Field(default=None, gt=0)
    val_size: int | None = Field(default=None, gt=0)
    model_name: str = Field(default="symptom-classifier")
    experiment_name: str = Field(default="dspy-symptom-classification")
    lm_model: str = Field(default="openai/gpt-4o-mini")
    num_threads: int = Field(default=4, gt=0)
    max_bootstrapped_demos: int = Field(default=3, ge=0)
    max_labeled_demos: int = Field(default=4, ge=0)

class TuneResponse(BaseModel):
    metrics: TuneMetrics
    model_info: ModelInfo
    mlflow_run_url: str

class TuneCommand:
    def __init__(self):
        config = self._build_config()
        self.service = ClassificationService(config)

    def execute(self, request: TuneRequest) -> TuneResponse:
        # Call service.tune()
        # Build response with run URL
```

**4.2: evaluate.py**

```python
class EvaluateRequest(BaseModel):
    model_name: str
    model_version: str | None = Field(default=None)
    split: Literal["train", "validation", "test"] = Field(default="test")

class EvaluateResponse(BaseModel):
    split: str
    accuracy: float
    num_examples: int
    metrics: dict[str, float]

class EvaluateCommand:
    def __init__(self):
        self.service = ClassificationService(self._build_config())

    def execute(self, request: EvaluateRequest) -> EvaluateResponse:
        # Call service.evaluate()
```

**4.3: list_models.py**

```python
class ListModelsRequest(BaseModel):
    name_filter: str | None = Field(default=None)

class ListModelsResponse(BaseModel):
    models: pd.DataFrame
    total_count: int
    model_config = {"arbitrary_types_allowed": True}

class ListModelsCommand:
    def __init__(self):
        self.service = ClassificationService(self._build_config())

    def execute(self, request: ListModelsRequest) -> ListModelsResponse:
        # Call service.list_models()
```

**Rationale:** Follow existing command pattern exactly (see `commands/dataset/list.py`). Pydantic validation, typed requests/responses, commands delegate to services.

---

### Step 5: Add CLI Interface
**File:** `src/symptom_diagnosis_explorer/cli.py`

**Changes:**

**1. Create classify sub-app:**
```python
classify_app = typer.Typer(help="Symptom classification with DSPy")
app.add_typer(classify_app, name="classify")
```

**2. Add tune command:**
```python
@classify_app.command(help="Tune/optimize classification model")
def tune(
    optimizer: OptimizerType = typer.Option("bootstrap", help="Optimizer type"),
    train_size: int | None = typer.Option(None, help="Limit training examples"),
    val_size: int | None = typer.Option(None, help="Limit validation examples"),
    model_name: str = typer.Option("symptom-classifier", help="Model name in registry"),
    experiment_name: str = typer.Option("dspy-symptom-classification", help="MLFlow experiment"),
    lm_model: str = typer.Option("openai/gpt-4o-mini", help="LLM model"),
    num_threads: int = typer.Option(4, help="Optimizer threads"),
    max_bootstrapped_demos: int = typer.Option(3, help="Max bootstrapped demos"),
    max_labeled_demos: int = typer.Option(4, help="Max labeled demos"),
):
    """
    Tune DSPy classification model with automatic validation evaluation.

    Examples:
        # Basic tuning with defaults
        symptom-diagnosis-explorer classify tune

        # Custom optimizer and dataset size
        symptom-diagnosis-explorer classify tune --optimizer mipro --train-size 100 --val-size 20

        # Different LLM model
        symptom-diagnosis-explorer classify tune --lm-model openai/gpt-4o
    """
    # Create request, execute command, display with Rich
```

**3. Add evaluate command:**
```python
@classify_app.command(help="Evaluate model on dataset split")
def evaluate(
    model_name: str = typer.Option("symptom-classifier", help="Model name"),
    model_version: str | None = typer.Option(None, help="Model version (latest if None)"),
    split: str = typer.Option("test", help="Dataset split: train/validation/test"),
):
    """
    Evaluate saved model on specified dataset split.

    Examples:
        # Evaluate latest version on test set
        symptom-diagnosis-explorer classify evaluate

        # Evaluate specific version on validation set
        symptom-diagnosis-explorer classify evaluate --model-version 2 --split validation
    """
```

**4. Add list-models command:**
```python
@classify_app.command(name="list-models", help="List registered models")
def list_models(
    name_filter: str | None = typer.Option(None, help="Filter by name pattern"),
):
    """
    List models in MLFlow registry.

    Examples:
        # List all models
        symptom-diagnosis-explorer classify list-models

        # Filter by name
        symptom-diagnosis-explorer classify list-models --name-filter symptom
    """
```

**Output Formatting:**
- Use Rich Console for tables (model lists, evaluation metrics)
- Use Rich Status for progress indicators during tuning
- Use Rich Panel for prediction results
- Follow existing patterns in `cli.py` dataset commands

**Rationale:** Consistent CLI UX with existing commands. Rich formatting for readability. Comprehensive help text for learning-focused spike.

---

### Step 6: Integration Tests

**6.1: Test Classification Service**
**File:** `tests/integration/services/test_classification.py` (new)

**Test Class:** `TestClassificationService`

**Fixtures:**
```python
@pytest.fixture
def classification_config():
    # Return test config with mock LLM or small real model

@pytest.fixture
def classification_service(classification_config):
    # Return configured service

@pytest.fixture
def small_dataset():
    # Return subset of dataset (5-10 examples)
```

**Tests:**
- `test_tune_with_bootstrap_optimizer`: Verify tuning completes, returns metrics
- `test_tune_logs_to_mlflow`: Check MLFlow run creation, artifacts
- `test_evaluate_returns_metrics`: Load model, evaluate, verify accuracy
- `test_predict_single_example`: Single prediction, verify diagnosis format
- `test_list_models_returns_dataframe`: Query registry, verify DataFrame structure
- `test_convert_to_dspy_examples`: DataFrame → DSPy Examples conversion

**Cleanup:**
- Delete MLFlow experiments after tests
- Clear DSPy cache if needed

**6.2: Test Classify Commands**
**File:** `tests/integration/commands/test_classify.py` (new)

**Test Class:** `TestClassifyCommands`

**Tests:**
- `test_tune_command_execution`: Create request, execute, verify response
- `test_evaluate_command_execution`: Execute with test split
- `test_predict_command_execution`: Execute with sample text
- `test_list_models_command_execution`: Verify response format

**Approach:** Similar to `test_dataset.py` - test command execution with real (small) data.

**Rationale:** Validate integration with external dependencies (DSPy, MLFlow). Follow existing test patterns. Keep tests fast with small datasets.

---

### Step 7: Configuration & Documentation

**7.1: Environment Variables**
Document in code/docstrings:
- `OPENAI_API_KEY`: Required for OpenAI models
- `DSPY_CACHEDIR`: Optional DSPy cache location

**7.2: MLFlow Configuration**
- `MLFLOW_TRACKING_URI`: Set as configuration setting (currently `.mlflow`)
  - This is a project setting, not an environment variable
  - Points to local file-based MLFlow tracking directory
  - Can be overridden for remote tracking server if needed

Add to command help text:
```bash
# MLFlow tracking is configured to use .mlflow directory by default
# To use a remote MLFlow server instead, update the configuration

# Start local MLFlow server (optional)
mlflow server --backend-store-uri sqlite:///mlflow.db

# Update configuration to point to server
# (Implementation detail: may use config file or environment variable override)
```

**7.3: Usage Examples**
Comprehensive examples in `--help` for each command (included in Step 5).

**7.4: Code Documentation**
- Docstrings for all classes/methods
- Type hints throughout
- Inline comments for complex DSPy/MLFlow interactions

**Rationale:** Learning-focused spike requires good documentation. Help future development and knowledge transfer.

---

## Implementation Order

1. **Step 2** (Models) - Foundation with no dependencies
2. **Step 1** (DatasetService) - Needed by service layer
3. **Step 3** (ClassificationService) - Core business logic
4. **Step 4** (Commands) - Depends on service
5. **Step 5** (CLI) - Depends on commands
6. **Step 6** (Tests) - Validate everything
7. **Step 7** (Documentation) - Throughout implementation

## Success Criteria

- [ ] Can run `symptom-diagnosis-explorer classify tune` and optimize model
- [ ] Tuning automatically evaluates on validation set
- [ ] MLFlow tracks experiments, metrics, and model artifacts
- [ ] Models saved to registry with "production" tag
- [ ] Can evaluate saved models on test set
- [ ] Can list registered models
- [ ] All commands work from Jupyter notebook
- [ ] Integration tests pass
- [ ] Code follows existing architectural patterns
- [ ] Comprehensive help documentation

## Out of Scope (Deferred)

- **Predict command** (Issue #4) - Single-shot prediction CLI and service method
- Production deployment configuration
- Advanced optimizer tuning
- Comprehensive evaluation metrics beyond accuracy
- Web API endpoints
- Authentication/security
- Extensive error handling for all edge cases
- Performance optimization
- Multiple LLM provider support (start with OpenAI)
