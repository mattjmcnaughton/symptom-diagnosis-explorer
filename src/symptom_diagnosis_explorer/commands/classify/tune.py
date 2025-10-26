"""Classify tune command."""

from pydantic import BaseModel, Field

from symptom_diagnosis_explorer.models.model_development import (
    ClassificationConfig,
    LMConfig,
    ModelInfo,
    OptimizerConfig,
    OptimizerType,
    TuneMetrics,
)
from symptom_diagnosis_explorer.services.model_development import (
    ModelDevelopmentService,
)


class TuneRequest(BaseModel):
    """Request model for tuning/optimizing classification model."""

    optimizer: OptimizerType = Field(
        default=OptimizerType.BOOTSTRAP_FEW_SHOT,
        description="Optimizer type to use",
    )
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
    num_threads: int = Field(
        default=4,
        gt=0,
        description="Number of parallel threads for optimization",
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001",
        description="MLFlow tracking server URI",
    )

    # Bootstrap-specific parameters
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

    # MIPROv2-specific parameters
    mipro_auto: str | None = Field(
        default="light",
        description="Auto mode for MIPROv2 ('light', 'medium', 'heavy', or None)",
    )
    mipro_minibatch_size: int = Field(
        default=35,
        gt=0,
        description="Minibatch size for MIPROv2 evaluation",
    )
    mipro_minibatch_full_eval_steps: int = Field(
        default=5,
        gt=0,
        description="Frequency of full validation evaluations in MIPROv2",
    )
    mipro_program_aware_proposer: bool = Field(
        default=True,
        description="Enable program-aware instruction generation in MIPROv2",
    )
    mipro_data_aware_proposer: bool = Field(
        default=True,
        description="Enable data-aware instruction generation in MIPROv2",
    )
    mipro_tip_aware_proposer: bool = Field(
        default=True,
        description="Enable tip-based instruction generation in MIPROv2",
    )
    mipro_fewshot_aware_proposer: bool = Field(
        default=True,
        description="Enable few-shot aware instruction generation in MIPROv2",
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
        # Build configuration from request
        config = ClassificationConfig(
            lm_config=LMConfig(model=request.lm_model),
            optimizer_config=OptimizerConfig(
                optimizer_type=request.optimizer,
                num_threads=request.num_threads,
                bootstrap_max_bootstrapped_demos=request.bootstrap_max_bootstrapped_demos,
                bootstrap_max_labeled_demos=request.bootstrap_max_labeled_demos,
                mipro_auto=request.mipro_auto,
                mipro_minibatch_size=request.mipro_minibatch_size,
                mipro_minibatch_full_eval_steps=request.mipro_minibatch_full_eval_steps,
                mipro_program_aware_proposer=request.mipro_program_aware_proposer,
                mipro_data_aware_proposer=request.mipro_data_aware_proposer,
                mipro_tip_aware_proposer=request.mipro_tip_aware_proposer,
                mipro_fewshot_aware_proposer=request.mipro_fewshot_aware_proposer,
            ),
            mlflow_experiment_name=request.experiment_name,
            mlflow_experiment_project=request.experiment_project,
            mlflow_tracking_uri=request.mlflow_tracking_uri,
        )
        self.service = ModelDevelopmentService(config)
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
