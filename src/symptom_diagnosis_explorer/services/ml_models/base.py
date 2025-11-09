"""Abstract base class for model development services."""

import mlflow
from abc import ABC, abstractmethod
from pathlib import Path

from symptom_diagnosis_explorer.models.model_development import (
    ClassificationConfig,
    EvaluateMetrics,
    ModelInfo,
    TuneMetrics,
)
from symptom_diagnosis_explorer.services.dataset import DatasetService


class BaseModelService(ABC):
    """Abstract base class for model development services.

    All framework-specific services (DSPy, LangChain, etc.) must extend this class
    and implement the required abstract methods. This ensures a consistent interface
    across all frameworks while allowing framework-specific implementations.

    The base class provides:
    - Common initialization pattern with config and dataset service
    - Shared MLFlow helper methods for logging dataset info
    - Abstract methods that must be implemented by subclasses

    Subclasses should:
    - Implement tune() and evaluate() methods
    - Implement the requires_training property
    - Call parent __init__ to initialize config and dataset service
    - Use the shared helper methods for consistency
    """

    def __init__(self, config: ClassificationConfig) -> None:
        """Initialize the model service with configuration.

        Args:
            config: Classification configuration (framework-agnostic fields).
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

        For trainable models (e.g., DSPy), this performs optimization/compilation.
        For non-trainable models (e.g., LangChain), this validates the configuration
        and evaluates on training/validation sets.

        Args:
            train_size: Optional limit on training examples.
            val_size: Optional limit on validation examples.
            model_name: Name for model registry.

        Returns:
            Tuple of (metrics, model_info) from tuning process.
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
        """Evaluate a saved model on specified dataset split.

        Args:
            model_name: Name of model in MLFlow registry.
            model_version: Specific version or None for latest.
            split: Dataset split to evaluate on (train, validation, test).
            eval_size: Optional limit on evaluation examples.

        Returns:
            Evaluation metrics including accuracy and run information.
        """
        pass

    @property
    @abstractmethod
    def requires_training(self) -> bool:
        """Whether this framework requires training/optimization.

        Returns:
            True for frameworks that require training (e.g., DSPy with compilation).
            False for frameworks that use fixed prompts (e.g., LangChain prompt engineering).
        """
        pass

    @property
    @abstractmethod
    def framework_type(self) -> str:
        """The framework type identifier for this service.

        Returns:
            Framework identifier string (e.g., "dspy", "langchain").
        """
        pass

    # Shared helper methods (concrete implementations)

    def _log_dataset_info(self, split: str, num_examples: int) -> None:
        """Log dataset information to MLFlow.

        This is a shared helper method used by all framework implementations
        to ensure consistent dataset logging.

        Args:
            split: Dataset split name (train, validation, test).
            num_examples: Number of examples in this split.
        """
        mlflow.log_param(f"dataset_{split}_source", self.config.dataset_identifier)
        mlflow.log_param(f"dataset_{split}_size", num_examples)

    def _setup_mlflow_experiment(self, framework_tag: str) -> None:
        """Setup MLFlow experiment with consistent tagging.

        This helper method sets up the MLFlow experiment and tags it appropriately
        for framework identification and filtering.

        Args:
            framework_tag: Framework identifier for tagging (e.g., "dspy", "langchain").
        """
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

        # Ensure MLFlow tracking directory structure exists
        tracking_uri = self.config.mlflow_tracking_uri
        if tracking_uri.startswith("file://"):
            tracking_dir = Path(tracking_uri.replace("file://", ""))
        else:
            tracking_dir = Path(tracking_uri)

        if tracking_dir.exists():
            trash_dir = tracking_dir / ".trash"
            trash_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_experiment(self.config.mlflow_experiment_name)
        mlflow.set_experiment_tags(
            {
                "system": "symptom-diagnosis-explorer",
                "project": self.config.mlflow_experiment_project,
                "framework": framework_tag,
                "trainable": str(self.requires_training).lower(),
            }
        )
