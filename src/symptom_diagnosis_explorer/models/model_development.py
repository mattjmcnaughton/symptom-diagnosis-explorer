"""Model development domain models for DSPy optimization and MLFlow tracking."""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, computed_field


class OptimizerType(str, Enum):
    """Enumeration of supported DSPy optimizer types."""

    BOOTSTRAP_FEW_SHOT = "bootstrap"
    MIPRO_V2 = "mipro"


class OptimizerConfig(BaseModel):
    """Configuration for DSPy optimizer/teleprompter.

    Attributes:
        optimizer_type: Which DSPy optimizer to use (BootstrapFewShot or MIPROv2).
        num_threads: Number of parallel threads for optimization.
        bootstrap_max_bootstrapped_demos: Maximum number of bootstrapped demonstrations to generate.
        bootstrap_max_labeled_demos: Maximum number of labeled demonstrations to use.
        mipro_auto: Auto mode for MIPROv2 ("light", "medium", "heavy", or None for manual).
        mipro_minibatch_size: Minibatch size for MIPROv2 evaluation.
        mipro_minibatch_full_eval_steps: Frequency of full validation set evaluations.
        mipro_program_aware_proposer: Enable program-aware instruction generation.
        mipro_data_aware_proposer: Enable data-aware instruction generation.
        mipro_tip_aware_proposer: Enable tip-based instruction generation.
        mipro_fewshot_aware_proposer: Enable few-shot aware instruction generation.
    """

    optimizer_type: OptimizerType = Field(
        default=OptimizerType.BOOTSTRAP_FEW_SHOT,
        description="Optimizer type to use",
    )
    num_threads: int = Field(
        default=4,
        gt=0,
        description="Number of parallel threads for optimization",
    )

    # Bootstrap-specific configuration
    bootstrap_max_bootstrapped_demos: int = Field(
        default=3,
        ge=0,
        description="Maximum number of bootstrapped demonstrations",
    )
    bootstrap_max_labeled_demos: int = Field(
        default=4,
        ge=0,
        description="Maximum number of labeled demonstrations",
    )

    # MIPROv2-specific configuration
    mipro_auto: str | None = Field(
        default="light",
        description=(
            "Auto mode for MIPROv2 optimization. "
            "'light': Fast tuning (6 candidates, 100 val samples) - good for quick iteration. "
            "'medium': Balanced (12 candidates, 300 val samples) - production baseline. "
            "'heavy': Thorough (18 candidates, 1000 val samples) - final optimization. "
            "None: Manual configuration of num_candidates and trials."
        ),
    )
    mipro_minibatch_size: int = Field(
        default=35,
        gt=0,
        description=(
            "Minibatch size for faster MIPROv2 evaluation. "
            "Reduces LM calls by evaluating on subsets. Lower = faster but less stable. "
            "Typical values: 20-50."
        ),
    )
    mipro_minibatch_full_eval_steps: int = Field(
        default=5,
        gt=0,
        description=(
            "Frequency of full validation set evaluations in MIPROv2. "
            "Higher values = fewer full evaluations = faster tuning. "
            "Typical values: 5-10."
        ),
    )
    mipro_program_aware_proposer: bool = Field(
        default=True,
        description=(
            "Enable program-aware instruction generation in MIPROv2. "
            "Analyzes program structure to generate better instructions. "
            "Disable for faster tuning with simpler prompts."
        ),
    )
    mipro_data_aware_proposer: bool = Field(
        default=True,
        description=(
            "Enable data-aware instruction generation in MIPROv2. "
            "Analyzes training data to generate domain-specific instructions. "
            "Disable for faster tuning with simpler prompts."
        ),
    )
    mipro_tip_aware_proposer: bool = Field(
        default=True,
        description=(
            "Enable tip-based instruction generation in MIPROv2. "
            "Uses optimization tips to guide instruction generation. "
            "Disable for faster tuning with simpler prompts."
        ),
    )
    mipro_fewshot_aware_proposer: bool = Field(
        default=True,
        description=(
            "Enable few-shot aware instruction generation in MIPROv2. "
            "Considers few-shot examples when generating instructions. "
            "Disable for faster tuning with simpler prompts."
        ),
    )


class LMConfig(BaseModel):
    """Configuration for language model.

    Attributes:
        model: Model identifier for task execution in format 'provider/model-name' (e.g., 'openai/gpt-4o-mini').
        prompt_model: Optional model identifier for prompt optimization. If None, uses the task model.
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random).
        max_tokens: Maximum tokens to generate in response.
    """

    model: str = Field(
        default="ollama/qwen3:1.7b",
        description="Language model for task execution",
    )
    prompt_model: str | None = Field(
        default=None,
        description="Language model for prompt optimization (None = use task model)",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=1_024,
        gt=0,
        description="Maximum tokens to generate",
    )


class FrameworkType(str, Enum):
    """Enumeration of supported ML frameworks."""

    DSPY = "dspy"
    LANGCHAIN = "langchain"
    PYDANTIC_AI = "pydantic-ai"


class BaseFrameworkConfig(BaseModel):
    """Base class for framework-specific configuration.

    All framework configs must extend this class and provide a framework
    type discriminator for Pydantic's discriminated union support.
    """

    framework: str = Field(description="Framework type identifier")

    @computed_field
    @property
    def requires_training(self) -> bool:
        """Whether this framework requires training/optimization.

        This should be overridden by subclasses to indicate if the framework
        requires a training phase (like DSPy compilation) or not (like LangChain
        prompt engineering).

        Returns:
            True if framework requires training, False otherwise.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement requires_training property"
        )


class DSPyConfig(BaseFrameworkConfig):
    """DSPy-specific framework configuration.

    DSPy requires training/optimization through compilation with optimizers
    like BootstrapFewShot or MIPROv2.
    """

    framework: Literal[FrameworkType.DSPY] = FrameworkType.DSPY
    optimizer_config: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="DSPy optimizer configuration",
    )

    @computed_field
    @property
    def requires_training(self) -> bool:
        """DSPy always requires training through compilation."""
        return True


class LangChainConfig(BaseFrameworkConfig):
    """LangChain-specific framework configuration.

    LangChain uses prompt engineering without training. The 'tune' operation
    validates the prompt and evaluates performance but does not optimize.

    Few-shot examples are managed directly in the prompt templates rather than
    being configured dynamically.
    """

    framework: Literal[FrameworkType.LANGCHAIN] = FrameworkType.LANGCHAIN
    prompt_template_name: str = Field(
        default="symptom_diagnosis",
        description="Name of the prompt template to use",
    )
    chain_type: Literal["simple", "structured_output"] = Field(
        default="structured_output",
        description="Type of LangChain chain to create",
    )

    @computed_field
    @property
    def requires_training(self) -> bool:
        """LangChain does not require training - it uses fixed prompts."""
        return False


class PydanticAIConfig(BaseFrameworkConfig):
    """Pydantic-AI-specific framework configuration.

    Pydantic-AI uses agent-based prompt engineering without training. The 'tune'
    operation validates the agent configuration and evaluates performance but does
    not optimize. Like LangChain, it uses fixed prompts.

    Key features:
    - Structured output validation with Pydantic models
    - Built-in retry mechanism with ModelRetry
    - Dynamic system prompts via decorators
    - Uses tool-based output mode (default, most reliable)
    """

    framework: Literal[FrameworkType.PYDANTIC_AI] = FrameworkType.PYDANTIC_AI
    num_few_shot_examples: int = Field(
        default=0,
        ge=0,
        description="Number of few-shot examples to include in system prompts",
    )

    @computed_field
    @property
    def requires_training(self) -> bool:
        """Pydantic-AI does not require training - it uses fixed agents."""
        return False


# Discriminated union of framework configs
FrameworkConfig = Annotated[
    Union[DSPyConfig, LangChainConfig, PydanticAIConfig],
    Field(discriminator="framework"),
]


class ClassificationConfig(BaseModel):
    """Top-level configuration for classification system.

    Supports multiple ML frameworks through a discriminated union.
    The framework_config field determines which framework-specific service
    will be used.

    Attributes:
        lm_config: Language model configuration (shared across frameworks).
        framework_config: Framework-specific configuration (DSPy or LangChain).
        optimizer_config: DEPRECATED - use framework_config.optimizer_config for DSPy.
        mlflow_experiment_name: Name of MLFlow experiment for tracking.
        mlflow_experiment_project: Project name for MLFlow experiment tagging.
        mlflow_tracking_uri: MLFlow tracking server URI.
        model_aliases: List of aliases to set when registering models.
        artifact_sample_size: Number of sample predictions to log as artifacts.
        dataset_identifier: HuggingFace dataset identifier for logging.
    """

    lm_config: LMConfig = Field(
        default_factory=LMConfig,
        description="Language model configuration",
    )
    framework_config: FrameworkConfig = Field(
        default_factory=lambda: DSPyConfig(),
        description="Framework-specific configuration (discriminated union)",
    )
    # DEPRECATED: kept for backward compatibility during migration
    optimizer_config: OptimizerConfig | None = Field(
        default=None,
        description="DEPRECATED - use framework_config.optimizer_config instead",
    )
    mlflow_experiment_name: str = Field(
        description="MLFlow experiment name (format: /symptom-diagnosis-explorer/{project}/{experiment})",
    )
    mlflow_experiment_project: str = Field(
        description="Project name for MLFlow experiment tagging",
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001",
        description="MLFlow tracking server URI",
    )
    mlflow_run_id: str | None = Field(
        default=None,
        description="MLFlow run ID to resume (None = create new runs)",
    )
    model_aliases: list[str] = Field(
        default=[],
        description="Aliases to set when registering models",
    )
    artifact_sample_size: int = Field(
        default=10,
        gt=0,
        description="Number of sample predictions to log as artifacts",
    )
    dataset_identifier: str = Field(
        default="gretelai/symptom_to_diagnosis",
        description="HuggingFace dataset identifier",
    )


class TuneMetrics(BaseModel):
    """Metrics from model tuning/optimization process.

    Attributes:
        train_accuracy: Accuracy on training set.
        validation_accuracy: Accuracy on validation set.
        num_train_examples: Number of training examples used.
        num_val_examples: Number of validation examples used.
    """

    train_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Training accuracy",
    )
    validation_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Validation accuracy",
    )
    num_train_examples: int = Field(
        gt=0,
        description="Number of training examples",
    )
    num_val_examples: int = Field(
        gt=0,
        description="Number of validation examples",
    )


class EvaluateMetrics(BaseModel):
    """Metrics from model evaluation.

    Attributes:
        accuracy: Accuracy on the evaluation set.
        num_examples: Number of examples evaluated.
        run_id: MLFlow run ID for this evaluation.
    """

    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Evaluation accuracy",
    )
    num_examples: int = Field(
        gt=0,
        description="Number of examples evaluated",
    )
    run_id: str = Field(description="MLFlow run ID")


class ModelInfo(BaseModel):
    """Metadata about a registered model.

    Attributes:
        name: Model name in MLFlow registry.
        version: Model version string.
        run_id: MLFlow run ID that produced this model.
        metrics: Dictionary of metric names to values.
    """

    name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    run_id: str = Field(description="MLFlow run ID")
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Model metrics",
    )
