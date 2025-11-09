"""Classify tune command."""

from pydantic import BaseModel, Field

from symptom_diagnosis_explorer.models.model_development import (
    ClassificationConfig,
    DSPyConfig,
    FrameworkType,
    LangChainConfig,
    LMConfig,
    ModelInfo,
    OptimizerConfig,
    OptimizerType,
    PydanticAIConfig,
    TuneMetrics,
)
from symptom_diagnosis_explorer.services.dataset import DatasetService
from symptom_diagnosis_explorer.services.ml_models import FrameworkRegistry


class TuneRequest(BaseModel):
    """Request model for tuning/optimizing classification model."""

    # Framework selection
    framework: FrameworkType = Field(
        default=FrameworkType.DSPY,
        description="ML framework to use (dspy or langchain)",
    )

    # Common parameters (all frameworks)
    train_size: int | None = Field(
        default=None,
        gt=0,
        description="Limit training examples (None = use all)",
    )
    val_size: int | None = Field(
        default=None,
        gt=0,
        description="Limit validation examples (None = use all)",
    )
    model_name: str = Field(
        default="symptom-classifier",
        description="Model name for MLFlow registry",
    )
    experiment_name: str = Field(
        description="MLFlow experiment name (auto-constructed as /symptom-diagnosis-explorer/{project}/{experiment-name})",
    )
    experiment_project: str = Field(
        description="MLFlow experiment project name for tagging",
    )
    lm_model: str = Field(
        default="ollama/qwen3:8b",
        description="Language model identifier",
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001",
        description="MLFlow tracking server URI",
    )

    # DSPy-specific parameters (prefixed with dspy_)
    dspy_optimizer: OptimizerType = Field(
        default=OptimizerType.BOOTSTRAP_FEW_SHOT,
        description="Optimizer type to use (DSPy only)",
    )
    dspy_num_threads: int = Field(
        default=4,
        gt=0,
        description="Number of parallel threads for optimization (DSPy only)",
    )
    dspy_bootstrap_max_bootstrapped_demos: int = Field(
        default=3,
        ge=0,
        description="Maximum number of bootstrapped demonstrations (DSPy Bootstrap only)",
    )
    dspy_bootstrap_max_labeled_demos: int = Field(
        default=4,
        ge=0,
        description="Maximum number of labeled demonstrations (DSPy Bootstrap only)",
    )
    dspy_mipro_auto: str | None = Field(
        default="light",
        description="Auto mode for MIPROv2 ('light', 'medium', 'heavy', or None) (DSPy MIPRO only)",
    )
    dspy_mipro_minibatch_size: int = Field(
        default=35,
        gt=0,
        description="Minibatch size for MIPROv2 evaluation (DSPy MIPRO only)",
    )
    dspy_mipro_minibatch_full_eval_steps: int = Field(
        default=5,
        gt=0,
        description="Frequency of full validation evaluations in MIPROv2 (DSPy MIPRO only)",
    )
    dspy_mipro_program_aware_proposer: bool = Field(
        default=True,
        description="Enable program-aware instruction generation in MIPROv2 (DSPy MIPRO only)",
    )
    dspy_mipro_data_aware_proposer: bool = Field(
        default=True,
        description="Enable data-aware instruction generation in MIPROv2 (DSPy MIPRO only)",
    )
    dspy_mipro_tip_aware_proposer: bool = Field(
        default=True,
        description="Enable tip-based instruction generation in MIPROv2 (DSPy MIPRO only)",
    )
    dspy_mipro_fewshot_aware_proposer: bool = Field(
        default=True,
        description="Enable few-shot aware instruction generation in MIPROv2 (DSPy MIPRO only)",
    )

    # LangChain-specific parameters (prefixed with langchain_)
    # Note: Few-shot examples are managed directly in the prompt templates
    # No additional parameters needed for LangChain at this time

    # Pydantic-AI-specific parameters (prefixed with pydantic_ai_)
    pydantic_ai_num_few_shot_examples: int = Field(
        default=0,
        ge=0,
        description="Number of few-shot examples (Pydantic-AI only)",
    )


class TuneResponse(BaseModel):
    """Response model for tune command."""

    metrics: TuneMetrics = Field(
        description="Training and validation metrics",
    )
    model_info: ModelInfo = Field(
        description="Registered model information",
    )
    run_id: str = Field(
        description="MLFlow run ID",
    )


class TuneCommand:
    """Command to tune/optimize classification model."""

    def __init__(self, request: TuneRequest) -> None:
        """Initialize the command.

        Args:
            request: Tune request with configuration parameters.
        """
        # Build framework-specific config based on selected framework
        if request.framework == FrameworkType.DSPY:
            framework_config = DSPyConfig(
                optimizer_config=OptimizerConfig(
                    optimizer_type=request.dspy_optimizer,
                    num_threads=request.dspy_num_threads,
                    bootstrap_max_bootstrapped_demos=request.dspy_bootstrap_max_bootstrapped_demos,
                    bootstrap_max_labeled_demos=request.dspy_bootstrap_max_labeled_demos,
                    mipro_auto=request.dspy_mipro_auto,
                    mipro_minibatch_size=request.dspy_mipro_minibatch_size,
                    mipro_minibatch_full_eval_steps=request.dspy_mipro_minibatch_full_eval_steps,
                    mipro_program_aware_proposer=request.dspy_mipro_program_aware_proposer,
                    mipro_data_aware_proposer=request.dspy_mipro_data_aware_proposer,
                    mipro_tip_aware_proposer=request.dspy_mipro_tip_aware_proposer,
                    mipro_fewshot_aware_proposer=request.dspy_mipro_fewshot_aware_proposer,
                ),
            )
        elif request.framework == FrameworkType.LANGCHAIN:
            framework_config = LangChainConfig()
        elif request.framework == FrameworkType.PYDANTIC_AI:
            framework_config = PydanticAIConfig(
                num_few_shot_examples=request.pydantic_ai_num_few_shot_examples,
            )
        else:
            raise ValueError(f"Unsupported framework: {request.framework}")

        # Build top-level configuration
        config = ClassificationConfig(
            lm_config=LMConfig(model=request.lm_model),
            framework_config=framework_config,
            mlflow_experiment_name=request.experiment_name,
            mlflow_experiment_project=request.experiment_project,
            mlflow_tracking_uri=request.mlflow_tracking_uri,
        )

        # Create service via registry with dataset service
        dataset_service = DatasetService()
        self.service = FrameworkRegistry.create_service(config, dataset_service)
        self.request = request

    def execute(self) -> TuneResponse:
        """Execute the tune command.

        Returns:
            Response containing metrics, model info, and run_id.

        Raises:
            RuntimeError: If the dataset cannot be loaded or tuning fails.
        """
        # Call service to tune model
        metrics, model_info = self.service.tune(
            train_size=self.request.train_size,
            val_size=self.request.val_size,
            model_name=self.request.model_name,
        )

        return TuneResponse(
            metrics=metrics,
            model_info=model_info,
            run_id=model_info.run_id,
        )
