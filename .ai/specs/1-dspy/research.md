# Research: Integrate DSPy for Simple Classification with MLFlow Tracking

## Issue Context
Issue #1: spike: integrate DSPy for simple classification with MLFlow tracking

This spike task focuses on integrating DSPy framework (a declarative framework for programming LLM applications) with MLFlow tracking for symptom-to-diagnosis classification. DSPy provides modular approaches to building LLM-based classifiers with automatic prompt optimization via optimizers (MIPROv2, BootstrapFewShot, etc.) and native MLFlow integration for experiment tracking and model versioning. The goal is to enhance diagnosis classification capabilities with structured, optimizable LLM pipelines that fit into the existing three-layer architecture (commands, services, models).

## Codebase Context

### Current Architecture

The symptom-diagnosis-explorer codebase follows a **clean three-layer architecture**:

1. **CLI/Commands Layer** (`src/symptom_diagnosis_explorer/cli.py`)
   - Uses Typer framework for modern CLI interfaces
   - Rich console output for formatted display
   - Sub-apps for grouping related commands (e.g., `dataset_app`)
   - Entry point: `symptom-diagnosis-explorer` command

2. **Commands Layer** (`src/symptom_diagnosis_explorer/commands/`)
   - **Command Pattern** with Pydantic Request/Response objects
   - Each command has: `*Request`, `*Response`, and `*Command` classes
   - Commands instantiate and use Service layer objects
   - Example structure:
     ```python
     class DatasetListRequest(BaseModel):
         split: str = Field(default="all")
         rows: int = Field(default=5, gt=0)

     class DatasetListResponse(BaseModel):
         df: pd.DataFrame
         total_rows: int
         model_config = {"arbitrary_types_allowed": True}

     class DatasetListCommand:
         def __init__(self):
             self.service = DatasetService()

         def execute(self, request: DatasetListRequest) -> DatasetListResponse:
             # Business logic
     ```

3. **Services Layer** (`src/symptom_diagnosis_explorer/services/`)
   - Contains business logic and external integrations
   - `DatasetService` manages Hugging Face dataset interactions
   - Lazy loading with caching pattern (stores in instance variables)
   - Methods return validated Pandera-typed DataFrames
   - Example methods: `load()`, `get_train_dataframe()`, `get_test_dataframe()`

4. **Models Layer** (`src/symptom_diagnosis_explorer/models/`)
   - **Domain models and data schemas**
   - `DiagnosisType` enum with 22 diagnosis categories (allergy, arthritis, etc.)
   - `SymptomDiagnosisExample` (Pydantic BaseModel):
     - Fields: `symptoms` (alias: input_text), `diagnosis` (alias: output_text)
     - Validators: whitespace stripping on symptoms
     - Property: `label` returns diagnosis for ML contexts
   - `SymptomDiagnosisDatasetDF` (Pandera schema):
     - DataFrame-level validation using Pandera
     - Validates each row as SymptomDiagnosisExample

### Existing Patterns to Follow

**Type Safety:**
- Pydantic for strict input/output validation
- Pandera for DataFrame schema validation
- Full Python type hints throughout
- Type checking with `ty` in pre-commit hooks

**Caching & Performance:**
- Lazy loading of datasets
- In-memory caching of loaded DataFrames
- Avoids redundant API calls

**Testing:**
- Class-based test organization with pytest
- Fixtures for setup (`@pytest.fixture`)
- Integration tests verify actual functionality
- Location: `/tests/integration/services/test_dataset.py` (265 lines)

**Code Quality:**
- Pre-commit hooks: Ruff linting/formatting, trailing whitespace, YAML validation
- Type checking via `ty check src`

### Dependencies Already Installed

The project already has all necessary dependencies in `pyproject.toml`:
- **DSPy** (>=3.0.3) - LLM programming framework
- **MLFlow** (>=3.5.0) - Experiment tracking
- **Transformers** (>=4.57.1) - Model fine-tuning support
- **PEFT** (>=0.17.1) - Parameter-efficient fine-tuning
- **Pandera[pandas,mypy]** (>=0.26.1) - Schema validation
- **Pydantic** (via dependencies) - Data validation
- **Typer** (>=0.20.0) - CLI framework
- **Rich** (>=14.0.0) - Formatted output
- **Datasets** (>=4.2.0) - Data loading
- **Pandas** (>=2.3.3) - Data manipulation

**Python Version:** >=3.13
**Build System:** Hatchling

### Recommended Integration Points for DSPy

Based on the existing architecture, natural integration points include:

1. **New Service Layer:** `services/classification.py` for DSPy classification logic
2. **New Models:** `models/classification.py` for DSPy signatures and configuration
3. **New Commands:** `commands/classify/` for classification CLI commands
4. **MLFlow Tracking:** Add MLFlow experiment tracking to service layer

## External Resources

### DSPy Framework Documentation

#### Official Tutorials
- **Classification Tutorial:** https://dspy.ai/tutorials/classification/
- **Classification Fine-tuning Tutorial:** https://dspy.ai/tutorials/classification_finetuning/
  - 77-way classification task using Llama-3.2-1B
  - Banking77 dataset example
  - Requires DSPy >= 2.6.0
  - Achieves 86.7% accuracy with 500 unlabeled examples
- **Real-World Examples:** https://dspy.ai/tutorials/real_world_examples/
- **Debugging & Observability:** https://dspy.ai/tutorials/observability/

#### DSPy Main Repository
- GitHub: https://github.com/stanfordnlp/dspy
- Framework for programming—not prompting—language models
- Stanford NLP lab project

#### Databricks Resources
- Build generative AI apps using DSPy on Databricks
- Notebook examples: https://notebooks.databricks.com/devrel/mlflow/2024-11-27-dspy.html
- Documentation: https://docs.databricks.com/aws/en/generative-ai/dspy/

#### Community Tutorials (2025)
- "DSPy Tutorial 2025: Build Better AI Systems with Automated Prompt Optimization" (Pondhouse Data)
- "Programming, Not Prompting: A Hands-On Guide to DSPy" (Towards Data Science)
- "Context Engineering — A Comprehensive Hands-On Tutorial with DSPy" (Towards Data Science)

### MLFlow Integration Documentation

#### Official MLFlow DSPy Integration
- **MLFlow DSPy Flavor:** https://mlflow.org/docs/latest/genai/flavors/dspy/
- **DSPy Optimizer Autologging:** https://mlflow.org/docs/latest/genai/flavors/dspy/optimizer/
- **Python API Reference:** https://mlflow.org/docs/latest/python_api/mlflow.dspy.html
- **Tracking DSPy Optimizers Tutorial:** https://dspy.ai/tutorials/optimizer_tracking/

#### Azure/Databricks Integration
- Tracing DSPy on Azure Databricks: https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/integrations/dspy
- Tracing DSPy on Databricks AWS: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/integrations/dspy

#### Talks & Presentations
- "Streamlining DSPy Development: Track, Debug, and Deploy With MLflow" (Data + AI Summit 2025)
- "Boosting DSPy Optimizer Observability with MLflow" (Medium article by AI on Databricks)

## Domain Knowledge

### DSPy Core Concepts

#### 1. Signatures
Signatures define the input/output schema for LLM interactions, enabling structured and type-safe LLM programming.

**Basic Classification Signature:**
```python
from typing import Literal
import dspy

class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""

    sentence: str = dspy.InputField()
    sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
    confidence: float = dspy.OutputField()
```

**Inline Signature (Shorthand):**
```python
classify = dspy.Predict('sentence -> sentiment: bool')
```

**Using Enum Types for Classification:**
```python
class Emotion(dspy.Signature):
    """Classify emotion."""

    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()
```

**With Instructions:**
```python
toxicity = dspy.Predict(
    dspy.Signature(
        "comment -> toxic: bool",
        instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks.",
    )
)
```

**Multi-field Classification:**
```python
signature = dspy.Signature("text, hint -> label").with_updated_fields(
    'label',
    type_=Literal[tuple(CLASSES)]
)
```

#### 2. Modules
Modules encapsulate LLM interactions and can be composed for complex workflows.

**Basic Predict Module:**
```python
classify = dspy.Predict(Classify)
response = classify(sentence="This book was super fun to read.")
```

**ChainOfThought Module:**
```python
classify = dspy.ChainOfThought(signature)
response = classify(text="What does a pending cash withdrawal mean?")
```

**Custom Module with Multiple Steps:**
```python
class ClassificationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ClassifySignature)

    def forward(self, text):
        return self.predict(text=text)
```

#### 3. Optimizers
Optimizers automatically improve prompts and model weights through systematic experimentation.

**BootstrapFinetune for Classification:**
```python
import dspy

optimizer = dspy.BootstrapFinetune(
    metric=(lambda x, y, trace=None: x.label == y.label),
    num_threads=24
)
optimized = optimizer.compile(classify, trainset=trainset)
```

**MIPROv2 Optimizer:**
```python
teleprompter = dspy.teleprompt.MIPROv2(
    metric=gsm8k_metric,
    auto="light"
)
optimized_program = teleprompter.compile(
    program,
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    requires_permission_to_run=False
)
```

#### 4. Datasets & Evaluation

**DataLoader for HuggingFace Datasets:**
```python
from dspy.datasets import DataLoader

trainset = [
    dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
    for x in DataLoader().from_huggingface(
        dataset_name="PolyAI/banking77",
        fields=("text", "label"),
        input_keys=("text",),
        split="train",
        trust_remote_code=True
    )[:2000]
]
```

**Evaluation API:**
```python
from dspy import Evaluate

evaluator = Evaluate(
    devset=devset,
    metric=classification_metric,
    num_threads=8
)
score = evaluator(optimized_program)
```

### MLFlow DSPy Integration Patterns

#### 1. Autologging Setup

**Basic Configuration:**
```python
import mlflow
import dspy

mlflow.dspy.autolog(
    log_traces=True,
    log_traces_from_compile=False,
    log_traces_from_eval=True,
    log_compiles=False,
    log_evals=False,
    disable=False,
    silent=False
)
```

**Full Tracking Configuration:**
```python
mlflow.dspy.autolog(
    log_compiles=True,           # Capture optimization process
    log_evals=True,              # Record evaluation results
    log_traces_from_compile=True # Track execution traces
)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy-Optimization")
```

#### 2. What Gets Tracked

MLFlow automatically logs:
- **Optimizer Hyperparameters:** Few-shot examples count, candidate numbers, settings
- **Program States:** Initial and optimized instructions, intermediate versions (JSON artifacts)
- **Datasets:** Training and evaluation data
- **Metrics:** Performance progression across evaluation steps
- **Traces:** Execution details, model responses, intermediate prompts

#### 3. Hierarchical Run Structure

When `log_compiles` and `log_evals` are enabled:
- **Parent Run:** Represents overall optimization process
  - Displays optimizer parameters
  - Shows overall metric progression
- **Child Runs:** Show each intermediate program version
  - Intermediate program states
  - Detailed execution traces

#### 4. Experiment Tracking Server

**Start MLFlow Server:**
```bash
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```
Access UI at: `http://127.0.0.1:5000/`

**Consolidating Runs:**
```python
with mlflow.start_run():
    # Optimization and evaluation here
    optimized_program = teleprompter.compile(program, trainset=trainset)
    score = evaluator(optimized_program, devset=devset)
```

#### 5. Configuration Options

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `log_traces` | true | Generate and log traces for program execution |
| `log_traces_from_compile` | false | Capture program execution during optimization |
| `log_traces_from_eval` | true | Capture program execution during evaluation |
| `log_compiles` | false | Create MLflow runs for each optimization cycle |
| `log_evals` | false | Create MLflow runs for each evaluation cycle |

### Performance Benchmarks

From official tutorials and community reports:
- GPT-4o-mini score improvement: 66% → 87% on classification tasks (DSPy 2.5.29)
- Banking77 dataset: 85.00/98 (86.7%) with 500 unlabeled examples
- BootstrapFinetune optimizer on tiny Llama-3.2-1B model

## Related Files & References

### Project Files
- `/goal.md` - Project goal and optimization targets
- `/pyproject.toml` - Dependencies and project configuration
- `/README.md` - Project overview
- `/src/symptom_diagnosis_explorer/models/diagnosis.py` - DiagnosisType enum (22 diagnoses)
- `/src/symptom_diagnosis_explorer/models/dataset.py` - SymptomDiagnosisExample, SymptomDiagnosisDatasetDF
- `/src/symptom_diagnosis_explorer/services/dataset.py` - DatasetService with caching
- `/src/symptom_diagnosis_explorer/commands/dataset/list.py` - Example command pattern
- `/tests/integration/services/test_dataset.py` - Testing patterns reference

### External Documentation
- DSPy GitHub: https://github.com/stanfordnlp/dspy
- DSPy Documentation: https://dspy.ai/
- MLFlow Documentation: https://mlflow.org/docs/latest/
- MLFlow DSPy Flavor: https://mlflow.org/docs/latest/genai/flavors/dspy/

## Key Considerations

### 1. Architecture Integration
- **Follow existing patterns:** Use Request/Response command pattern, Service layer for business logic, Pydantic models for validation
- **Maintain separation of concerns:** DSPy signatures in models/, classification logic in services/, CLI commands in commands/
- **Type safety:** Ensure DSPy signatures work with Pydantic validation and Pandera schemas
- **Caching:** Consider caching compiled DSPy programs to avoid recompilation

### 2. DSPy-Specific Considerations
- **Signature design:** Map existing `SymptomDiagnosisExample` to DSPy signature with `symptoms: str` input and `diagnosis: Literal[DiagnosisType]` output
- **Module selection:** Start with `dspy.Predict` or `dspy.ChainOfThought` for basic classification
- **Optimizer choice:** Consider `BootstrapFewShot` for quick testing, `MIPROv2` for more sophisticated optimization
- **LLM Configuration:** Need to configure `dspy.LM()` with appropriate model (e.g., `openai/gpt-4o-mini`)
- **Dataset format:** Convert existing Pandera DataFrames to DSPy `Example` objects

### 3. MLFlow Integration
- **Tracking server:** Need to start MLFlow tracking server locally or use remote server
- **Experiment organization:** Create separate experiments for different optimization approaches
- **Autologging configuration:** Enable `log_compiles=True`, `log_evals=True`, `log_traces_from_compile=True` for comprehensive tracking
- **Run management:** Use `mlflow.start_run()` contexts to organize related operations
- **Artifact storage:** Consider where to store optimized program artifacts

### 4. Evaluation Metrics
- **Compatibility:** Create metrics compatible with both DSPy's `Evaluate` API and existing evaluation patterns
- **Accuracy metric:** Simple `lambda x, y: x.diagnosis == y.diagnosis` for classification accuracy
- **Multi-class support:** DSPy supports Literal types for multi-class classification (22 diagnosis types)
- **Confidence scoring:** Consider adding confidence outputs to signatures

### 5. Testing Strategy
- **Unit tests:** Test DSPy signature definitions and module instantiation
- **Integration tests:** Test classification service with real datasets and MLFlow tracking
- **Fixtures:** Create pytest fixtures for DSPy modules, datasets, and MLFlow experiments
- **Mocking:** Consider mocking LLM calls for faster unit tests
- **Cleanup:** Ensure MLFlow experiments and runs are cleaned up after tests

### 6. CLI Interface Design
- **Commands to add:**
  - `symptom-diagnosis-explorer classify run --text "symptoms" --model "model-name"`
  - `symptom-diagnosis-explorer classify optimize --optimizer mipro --experiments 10`
  - `symptom-diagnosis-explorer classify evaluate --model-path "path/to/model"`
- **Output format:** Use Rich console for formatted output (existing pattern)
- **Error handling:** Graceful handling of LLM API failures, invalid inputs

### 7. Configuration Management
- **LLM credentials:** Need secure handling of API keys (environment variables, config files)
- **MLFlow server URI:** Configurable tracking server URI
- **Model selection:** Support multiple LLM providers (OpenAI, Anthropic, local models)
- **Optimizer settings:** Configurable optimizer hyperparameters

### 8. Documentation Needs
- **Usage examples:** How to run classification, optimization, evaluation
- **DSPy patterns:** Document signature design patterns for this project
- **MLFlow setup:** How to start tracking server, view experiments
- **Configuration:** How to configure LLM credentials, MLFlow URI
- **Troubleshooting:** Common errors and solutions

### 9. Performance & Resource Considerations
- **LLM API costs:** Optimization can make many LLM calls (consider rate limits, costs)
- **Computation time:** BootstrapFinetune with many examples can be slow
- **Memory usage:** Caching compiled programs and datasets
- **Concurrent execution:** DSPy optimizers support `num_threads` parameter

### 10. Scope Management (Spike Task)
- **Focus on learning:** Prioritize understanding DSPy patterns over production-readiness
- **Simple examples first:** Start with basic `dspy.Predict` before complex optimizers
- **Document learnings:** Capture architectural insights, gotchas, best practices
- **Validate approach:** Ensure DSPy + MLFlow fit well with existing architecture
- **Minimal dataset:** Use small subset for faster iteration during spike
