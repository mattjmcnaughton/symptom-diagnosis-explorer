"""Classify evaluate command."""

from typing import Literal

from pydantic import BaseModel, Field

from symptom_diagnosis_explorer.models.model_development import (
    ClassificationConfig,
    DSPyConfig,
    EvaluateMetrics,
    FrameworkType,
    LangChainConfig,
    PydanticAIConfig,
)
from symptom_diagnosis_explorer.services.dataset import DatasetService
from symptom_diagnosis_explorer.services.ml_models import FrameworkRegistry


class EvaluateRequest(BaseModel):
    """Request model for evaluating a saved model."""

    # Framework selection
    framework: FrameworkType = Field(
        default=FrameworkType.DSPY,
        description="ML framework to use (dspy or langchain)",
    )

    # Common parameters
    model_name: str = Field(
        default="symptom-classifier",
        description="Model name in MLFlow registry",
    )
    model_version: str | None = Field(
        default=None,
        description="Model version (None = latest)",
    )
    split: Literal["train", "validation", "test"] = Field(
        default="test",
        description="Dataset split to evaluate on",
    )
    eval_size: int | None = Field(
        default=None,
        gt=0,
        description="Limit evaluation examples (None = use all)",
    )
    experiment_name: str = Field(
        default="/symptom-diagnosis-explorer/default/evaluate",
        description="MLFlow experiment name for evaluation tracking",
    )
    experiment_project: str = Field(
        default="default",
        description="MLFlow experiment project name for tagging",
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001",
        description="MLFlow tracking server URI",
    )

    # LangChain-specific parameters
    # Note: Few-shot examples are managed directly in the prompt templates
    # No additional parameters needed for LangChain at this time

    # Pydantic-AI-specific parameters
    # Note: Agent is recreated from hardcoded configuration
    # No additional parameters needed for Pydantic-AI at this time


class EvaluateResponse(BaseModel):
    """Response model for evaluate command."""

    split: str = Field(
        description="Dataset split evaluated",
    )
    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Accuracy on the evaluation set",
    )
    num_examples: int = Field(
        gt=0,
        description="Number of examples evaluated",
    )
    run_id: str = Field(
        description="MLFlow run ID for this evaluation",
    )


class EvaluateCommand:
    """Command to evaluate a saved model on a dataset split."""

    def execute(self, request: EvaluateRequest) -> EvaluateResponse:
        """Execute the evaluate command.

        Args:
            request: Evaluate request with model and split parameters.

        Returns:
            Response containing evaluation metrics.

        Raises:
            RuntimeError: If the dataset cannot be loaded.
            ValueError: If the split is invalid.
        """
        # Build framework-specific config based on selected framework
        if request.framework == FrameworkType.DSPY:
            # DSPy doesn't need framework_config for evaluation (loads from MLFlow)
            framework_config = DSPyConfig()
        elif request.framework == FrameworkType.LANGCHAIN:
            # LangChain recreates chain from hardcoded prompts
            framework_config = LangChainConfig()
        elif request.framework == FrameworkType.PYDANTIC_AI:
            # Pydantic-AI recreates agent from hardcoded configuration
            framework_config = PydanticAIConfig()
        else:
            raise ValueError(f"Unsupported framework: {request.framework}")

        # Create configuration for evaluation with experiment tracking
        config = ClassificationConfig(
            framework_config=framework_config,
            mlflow_experiment_name=request.experiment_name,
            mlflow_experiment_project=request.experiment_project,
            mlflow_tracking_uri=request.mlflow_tracking_uri,
        )

        # Create service via registry with dataset service
        dataset_service = DatasetService()
        service = FrameworkRegistry.create_service(config, dataset_service)

        # Call service to evaluate model
        metrics: EvaluateMetrics = service.evaluate(
            model_name=request.model_name,
            model_version=request.model_version,
            split=request.split,
            eval_size=request.eval_size,
        )

        return EvaluateResponse(
            split=request.split,
            accuracy=metrics.accuracy,
            num_examples=metrics.num_examples,
            run_id=metrics.run_id,
        )
