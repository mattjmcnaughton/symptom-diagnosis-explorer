# Plan: Add Pydantic-AI Framework Support

## Step 1: Write Plan to plan.md
Write this complete implementation plan to `plan.md` in the project root directory before starting any implementation work.

## Overview
Add pydantic-ai framework support following the same pattern as LangChain, creating a non-trainable classification service that uses Ollama models with structured outputs. Like LangChain, we will NOT serialize/save models but WILL use MLFlow autologging for traces and metrics.

## System Prompt Design
Create a medical diagnostic assistant prompt emphasizing:
- Expert symptom-to-diagnosis classification
- Structured output with Pydantic validation
- Valid diagnosis categories from DiagnosisType enum
- Clear instructions for exact matching
- Few-shot examples support via dynamic system prompts (using `@agent.system_prompt` decorators)

## Implementation Steps

### Step 2: Add PydanticAIConfig to model_development.py
- Add `PYDANTIC_AI = "pydantic-ai"` to `FrameworkType` enum
- Create `PydanticAIConfig(BaseFrameworkConfig)` class with:
  - `framework: Literal[FrameworkType.PYDANTIC_AI]`
  - `output_mode: Literal["tool", "native", "prompted"] = "native"`
  - `num_few_shot_examples: int = 0`
  - `@computed_field requires_training: bool = False` property
- Add `PydanticAIConfig` to `FrameworkConfig` discriminated union

### Step 3: Create services/ml_models/pydantic_ai.py
Main implementation file following LangChain's pattern:

**Imports:**
- `pydantic_ai`: `Agent`, `RunContext`, `ModelRetry`
- Standard: `json`, `mlflow`, `pandas`, `Path`, `BaseModel`, `Field`

**Structured Output:**
```python
class DiagnosisOutput(BaseModel):
    """Structured output schema for diagnosis predictions."""
    diagnosis: str = Field(description="The predicted diagnosis")
```

**Agent Creation Helper:**
```python
def _get_system_instructions(diagnosis_types: list[str]) -> str:
    """Generate static system instructions with diagnosis list."""
    # Similar to LangChain's prompt but adapted for pydantic-ai style
    # Emphasizes: return ONLY diagnosis name, exact match required
```

**PydanticAIModelService(BaseModelService):**

- **`__init__()`**:
  - Validate `config.framework_config` is `PydanticAIConfig`
  - Setup MLFlow experiment via `_setup_mlflow_experiment("pydantic-ai")`
  - **Enable MLFlow autologging** (like LangChain):
    ```python
    mlflow.pydantic_ai.autolog(
        log_traces=True,  # Enable execution traces
        disable=False,
        silent=False,
    )
    ```

- **`requires_training`** property: `return False`

- **`framework_type`** property: `return FrameworkType.PYDANTIC_AI`

- **`_create_classification_agent()`**:
  ```python
  # Extract model name from "ollama/model-name" format
  model_name = self.config.lm_config.model.replace("ollama/", "")

  # Create agent with structured output
  agent = Agent(
      f'ollama:{model_name}',
      output_type=DiagnosisOutput,
      instructions=_get_system_instructions(diagnosis_types),
  )

  # Add dynamic system prompts for few-shot examples
  @agent.system_prompt
  def add_few_shot_examples(ctx: RunContext) -> str:
      if self.pydantic_ai_config.num_few_shot_examples > 0:
          # Return few-shot examples
      return ""

  # Add output validator for diagnosis validation
  @agent.output_validator
  def validate_diagnosis(ctx: RunContext, output: DiagnosisOutput) -> DiagnosisOutput:
      if output.diagnosis.strip().lower() not in [dt.lower() for dt in diagnosis_types]:
          raise ModelRetry(f'Invalid diagnosis. Must be one of: {diagnosis_types}')
      return output

  return agent
  ```

- **`_evaluate_on_dataset()`**:
  - Run agent synchronously on each row: `result = agent.run_sync(row["symptoms"])`
  - Extract diagnosis from `result.output.diagnosis`
  - Calculate accuracy with case-insensitive comparison
  - Log metrics, dataset info, and sample predictions (like LangChain)

- **`tune()`**:
  - Load train/val datasets
  - Create agent via `_create_classification_agent()`
  - Start MLFlow run
  - Log parameters (framework, model, prompt details, dataset sizes)
  - Evaluate on both train and val splits
  - **Log config as JSON artifact** (NOT the agent itself)
  - Create registered model and version (points to artifact URI)
  - Set model aliases
  - Return `TuneMetrics` and `ModelInfo`

- **`evaluate()`**:
  - Load specified split (train/val/test)
  - **Recreate agent from current config** (NOT from MLFlow)
  - Evaluate on split with `_evaluate_on_dataset()`
  - Log parameters and metrics
  - Return `EvaluateMetrics`

- **`list_models()`**:
  - Query MLFlow registry (reuse pattern from LangChain)

### Step 4: Update services/ml_models/registry.py
- Register `PydanticAIModelService` with decorator:
  ```python
  @FrameworkRegistry.register(FrameworkType.PYDANTIC_AI)
  class PydanticAIModelService(BaseModelService):
  ```

### Step 5: Update services/ml_models/__init__.py
- Import `PydanticAIModelService`
- Add to `__all__` list

### Step 6: Update commands/classify/tune.py
- Add to `TuneRequest`:
  ```python
  # Pydantic-AI specific parameters (prefixed with pydantic_ai_)
  pydantic_ai_output_mode: Literal["tool", "native", "prompted"] = Field(
      default="native",
      description="Structured output mode (Pydantic-AI only)",
  )
  pydantic_ai_num_few_shot_examples: int = Field(
      default=0,
      ge=0,
      description="Number of few-shot examples (Pydantic-AI only)",
  )
  ```

- Add framework config creation in `TuneCommand.__init__()`:
  ```python
  elif request.framework == FrameworkType.PYDANTIC_AI:
      framework_config = PydanticAIConfig(
          output_mode=request.pydantic_ai_output_mode,
          num_few_shot_examples=request.pydantic_ai_num_few_shot_examples,
      )
  ```

### Step 7: Update commands/classify/evaluate.py
- Add similar `elif request.framework == FrameworkType.PYDANTIC_AI:` block
- Add parameters to `EvaluateRequest` if needed

### Step 8: Update cli.py
- Add CLI options:
  ```python
  pydantic_ai_output_mode: Annotated[
      str,
      typer.Option(help="[Pydantic-AI] Structured output mode (tool/native/prompted)"),
  ] = "native"

  pydantic_ai_num_few_shot_examples: Annotated[
      int,
      typer.Option(help="[Pydantic-AI] Number of few-shot examples"),
  ] = 0
  ```
- Pass to TuneRequest/EvaluateRequest

### Step 9: Create tests/integration/services/ml_models/test_pydantic_ai_integration.py
Full test suite mirroring LangChain tests:

- Module-level Ollama model check: `REQUIRED_MODELS = ["qwen3:1.7b"]`

- Fixtures:
  ```python
  @pytest.fixture
  def pydantic_ai_config_zero_shot(test_model_name, mlflow_test_dir) -> ClassificationConfig

  @pytest.fixture
  def pydantic_ai_config_few_shot(test_model_name, mlflow_test_dir) -> ClassificationConfig

  @pytest.fixture
  def pydantic_ai_service_zero_shot(pydantic_ai_config_zero_shot) -> PydanticAIModelService

  @pytest.fixture
  def pydantic_ai_service_few_shot(pydantic_ai_config_few_shot) -> PydanticAIModelService
  ```

- Test cases:
  - `test_tune_zero_shot_with_small_dataset()`: 5 train, 3 val
  - `test_tune_few_shot_with_small_dataset()`: 10 train, 3 val
  - `test_evaluate_on_test_split()`: Tune then evaluate
  - `test_evaluate_latest_version()`: Use latest version
  - `test_list_models()`: Create multiple models, verify listing
  - `test_requires_training_property()`: Assert False
  - `test_framework_type_property()`: Assert "pydantic-ai"
  - `test_invalid_config_type_raises_error()`: Pass DSPyConfig to PydanticAI service

- Markers: `@pytest.mark.integration`, `@pytest.mark.llm`, `@pytest.mark.ollama`, `@pytest.mark.slow`

### Step 10: Create notebooks/projects/9-langchain/experiments/2025-11-11-pydantic-ai-pipeline.ipynb
Notebook structure (sibling to langchain notebook):

**Cells:**
1. **Overview** (markdown): Explain pydantic-ai approach, key differences from DSPy/LangChain
2. **Imports**: Import TuneCommand, EvaluateCommand, FrameworkType
3. **Config**: Set PROJECT, EXPERIMENT_NAME, LM_MODEL, sizes (same as langchain notebook)
4. **Tune**: Run pydantic-ai tuning with zero-shot
5. **Tune Results**: Display train/val accuracy, model info
6. **Evaluate**: Run evaluation on test set
7. **Evaluate Results**: Display test accuracy

**Configuration:**
- PROJECT: `"9-langchain"` (share same project for comparison)
- EXPERIMENT_NAME: `"pydantic-ai-pipeline"`
- LM_MODEL: `"ollama/qwen3:0.6b"` (same as langchain)
- TRAIN_SIZE: 15, VAL_SIZE: 20, TEST_SIZE: 10 (same as langchain)

### Step 11: Create notebooks/projects/9-langchain/experiments/2025-11-11-framework-comparison.ipynb
Comparison notebook structure:

**Cells:**
1. **Overview** (markdown): Compare LangChain vs Pydantic-AI on same task/data
2. **Imports**: Import both TuneCommand and necessary libs
3. **Config**: Shared config for both frameworks
4. **Run LangChain**: Tune + evaluate
5. **Run Pydantic-AI**: Tune + evaluate
6. **Compare Results** (code + markdown):
   - Create comparison DataFrame with metrics side-by-side
   - Show train/val/test accuracy for both
   - Compare MLFlow trace counts, token usage if available
   - Discuss tradeoffs (type safety, API simplicity, validation approach)
7. **Conclusions** (markdown):
   - Which framework for which use case
   - Performance characteristics
   - Developer experience notes

### Step 12: Update CLAUDE.md documentation
Add pydantic-ai to relevant sections:

**Framework Characteristics Table:**
```
| Framework   | Requires Training | MLFlow Usage | Evaluation Strategy |
|-------------|-------------------|--------------|---------------------|
| DSPy        | Yes               | Loads compiled model | Uses saved optimization artifacts |
| LangChain   | No                | Only logs metrics/params | Recreates chain from hardcoded prompts |
| Pydantic-AI | No                | Only logs metrics/params | Recreates agent from config |
```

**CLI Usage Examples:**
```bash
# Tune Pydantic-AI model (zero-shot)
symptom-diagnosis-explorer classify tune \
    --framework pydantic-ai \
    --train-size 100 \
    --val-size 20

# Tune Pydantic-AI model (few-shot with native output mode)
symptom-diagnosis-explorer classify tune \
    --framework pydantic-ai \
    --pydantic-ai-output-mode native \
    --pydantic-ai-num-few-shot-examples 3 \
    --train-size 100 \
    --val-size 20

# Evaluate Pydantic-AI model
symptom-diagnosis-explorer classify evaluate \
    --framework pydantic-ai \
    --model-name symptom-classifier \
    --split test
```

## Key Design Decisions

1. **No Model Serialization**: Like LangChain, pydantic-ai does NOT serialize agents to MLFlow
   - Only log configuration as JSON artifact
   - Recreate agent from config during evaluation
   - Avoids MLFlow integration complexity

2. **MLFlow Autologging**: Enable `mlflow.pydantic_ai.autolog()` for:
   - Execution traces with token usage
   - Automatic span tracking
   - Performance metrics

3. **Output Mode**: Default to `"native"` for most reliable structured output with Ollama
   - More reliable than `"tool"` or `"prompted"` modes
   - Uses model's native structured output capabilities

4. **Few-Shot Support**: Use `@agent.system_prompt` decorators
   - Cleaner than hardcoded templates
   - Dynamic context injection via RunContext
   - More maintainable than LangChain's approach

5. **Validation**: Use `@agent.output_validator` with `ModelRetry`
   - Automatic retries on invalid diagnoses
   - Built-in error handling (better than manual checking)
   - Leverages pydantic-ai's retry mechanism

6. **Ollama Integration**: Use string format `'ollama:model-name'`
   - Simplest approach, no custom provider setup
   - Extract from `ollama/model-name` config format

## Dependencies
- ✅ `pydantic-ai-slim[openai]>=1.14.1` already in pyproject.toml
- No additional dependencies needed

## Testing Strategy
- Use same Ollama model (`qwen3:1.7b`) as LangChain tests
- Small dataset sizes (5-10 examples) for speed
- Verify all integration points work end-to-end
- Test both zero-shot and few-shot configurations
- Ensure MLFlow autologging captures traces

## Benefits vs LangChain
- **Better type safety**: Full Pydantic validation throughout
- **Simpler validation**: Built-in `ModelRetry` vs manual checking
- **Cleaner prompts**: Decorator-based vs template strings
- **Better IDE support**: Full type hints and autocomplete
- **More reliable structured output**: Native mode support
- **Better error handling**: Automatic retries with ModelRetry

## Deliverables Summary
1. ✅ Plan document (plan.md) - **FIRST STEP**
2. Framework config model (PydanticAIConfig)
3. Service implementation (pydantic_ai.py)
4. Registry integration
5. Command layer updates (tune.py, evaluate.py)
6. CLI updates
7. Integration tests
8. Experiment notebook (pydantic-ai-pipeline.ipynb)
9. Comparison notebook (framework-comparison.ipynb)
10. Documentation updates (CLAUDE.md)
