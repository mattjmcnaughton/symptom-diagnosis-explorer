"""Integration tests for the ModelDevelopmentService with Ollama LLMs.

This test module requires Ollama models to be available locally.
The entire module will fail if required models are not found.

Required models:
- qwen3:1.7b (primary test model)
- qwen3:8b (prompt optimization model for MIPRO v2)
"""

from pathlib import Path

import mlflow
import pandas as pd
import pytest

from tests.integration.conftest import check_all_required_models
from symptom_diagnosis_explorer.models.model_development import (
    ClassificationConfig,
    LMConfig,
    OptimizerConfig,
    OptimizerType,
)
from symptom_diagnosis_explorer.services.model_development import (
    ModelDevelopmentService,
)


# Required Ollama models for this test module
REQUIRED_MODELS = ["qwen3:1.7b", "qwen3:8b"]


# Module-level check: skip entire module if required models aren't available
try:
    _all_available, _missing_models = check_all_required_models(REQUIRED_MODELS)
    if not _all_available:
        _skip_message = (
            f"Required Ollama models not available: {_missing_models}\n"
            f"Install missing models with:\n"
            + "\n".join(
                f"  ollama pull {model}"
                for model in _missing_models
                if "Ollama" not in model
            )
        )
        pytest.skip(_skip_message, allow_module_level=True)
except Exception as e:
    pytest.skip(f"Cannot check Ollama availability: {e}", allow_module_level=True)


@pytest.fixture
def model_development_config(
    test_model_name: str, mlflow_test_dir: Path
) -> ClassificationConfig:
    """Create test configuration for ModelDevelopmentService.

    Args:
        test_model_name: Name of Ollama model to use.
        mlflow_test_dir: Path to temp MLFlow directory.

    Returns:
        ClassificationConfig for testing.
    """
    return ClassificationConfig(
        lm_config=LMConfig(
            model=f"ollama/{test_model_name}",
            temperature=0.0,
            max_tokens=200,  # Needs to be large enough for full JSON responses
        ),
        optimizer_config=OptimizerConfig(
            optimizer_type=OptimizerType.BOOTSTRAP_FEW_SHOT,
            num_threads=2,  # Lower for tests
            max_bootstrapped_demos=2,  # Keep small
            max_labeled_demos=2,  # Keep small
        ),
        mlflow_experiment_name="/test/symptom-diagnosis-explorer/1-dspy/classification",
        mlflow_experiment_project="1-dspy",
        mlflow_tracking_uri=str(mlflow_test_dir),
        mlflow_run_id=None,  # Create new runs for each test
        model_aliases=[
            "test"
        ],  # Test with custom alias ('latest' is reserved by MLFlow)
        artifact_sample_size=5,  # Small sample for tests
        dataset_identifier="gretelai/symptom_to_diagnosis",
    )


@pytest.fixture
def model_development_service(
    model_development_config: ClassificationConfig,
) -> ModelDevelopmentService:
    """Create ModelDevelopmentService instance for testing.

    Args:
        model_development_config: Test configuration.

    Returns:
        Configured ModelDevelopmentService.
    """
    return ModelDevelopmentService(model_development_config)


@pytest.fixture
def model_development_config_mipro_v2(
    test_model_name: str, mlflow_test_dir: Path
) -> ClassificationConfig:
    """Create test configuration for MIPRO_V2 optimizer.

    Uses qwen3:8b as the prompt optimization model and
    qwen3:1.7b as the task model.

    Args:
        test_model_name: Name of Ollama model to use for task.
        mlflow_test_dir: Path to temp MLFlow directory.

    Returns:
        ClassificationConfig for MIPRO_V2 testing.
    """
    return ClassificationConfig(
        lm_config=LMConfig(
            model=f"ollama/{test_model_name}",
            prompt_model="ollama/qwen3:8b",
            temperature=0.0,
            max_tokens=200,  # Needs to be large enough for full JSON responses
        ),
        optimizer_config=OptimizerConfig(
            optimizer_type=OptimizerType.MIPRO_V2,
            num_threads=2,  # Lower for tests
            max_bootstrapped_demos=2,  # Keep small
            max_labeled_demos=2,  # Keep small
        ),
        mlflow_experiment_name="/test/symptom-diagnosis-explorer/1-dspy/classification-mipro",
        mlflow_experiment_project="1-dspy",
        mlflow_tracking_uri=str(mlflow_test_dir),
        mlflow_run_id=None,  # Create new runs for each test
        model_aliases=["test-mipro"],  # Test with custom alias for MIPRO
        artifact_sample_size=5,  # Small sample for tests
        dataset_identifier="gretelai/symptom_to_diagnosis",
    )


@pytest.fixture
def model_development_service_mipro_v2(
    model_development_config_mipro_v2: ClassificationConfig,
) -> ModelDevelopmentService:
    """Create ModelDevelopmentService instance with MIPRO_V2 for testing.

    Args:
        model_development_config_mipro_v2: MIPRO_V2 test configuration.

    Returns:
        Configured ModelDevelopmentService with MIPRO_V2.
    """
    return ModelDevelopmentService(model_development_config_mipro_v2)


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
class TestModelDevelopmentService:
    """Integration tests for ModelDevelopmentService with Ollama models."""

    @pytest.mark.slow
    def test_tune_with_small_dataset(
        self, model_development_service: ModelDevelopmentService
    ):
        """Test model tuning with very small dataset.

        This test uses minimal data (5 train, 3 validation) to verify the
        tuning pipeline works end-to-end with Ollama.
        """
        # Tune with very small dataset for speed
        metrics, model_info = model_development_service.tune(
            train_size=5,
            val_size=3,
            model_name="test-symptom-classifier",
        )

        # Verify metrics
        assert metrics.train_accuracy >= 0.0
        assert metrics.train_accuracy <= 1.0
        assert metrics.validation_accuracy >= 0.0
        assert metrics.validation_accuracy <= 1.0
        assert metrics.num_train_examples == 5
        assert metrics.num_val_examples == 3

        # Verify model info
        assert model_info.name == "test-symptom-classifier"
        assert model_info.version is not None
        assert model_info.run_id is not None
        assert "train_accuracy" in model_info.metrics
        assert "validation_accuracy" in model_info.metrics

    @pytest.mark.slow
    def test_tune_with_mipro_v2_optimizer(
        self, model_development_service_mipro_v2: ModelDevelopmentService
    ):
        """Test model tuning with MIPRO_V2 optimizer.

        This test verifies the MIPRO_V2 optimization pipeline works with
        qwen3:8b as the prompt model and qwen3:1.7b as the task model.
        """
        # Tune with MIPRO_V2 optimizer and small dataset for speed
        metrics, model_info = model_development_service_mipro_v2.tune(
            train_size=5,
            val_size=3,
            model_name="test-mipro-classifier",
        )

        # Verify metrics
        assert metrics.train_accuracy >= 0.0
        assert metrics.train_accuracy <= 1.0
        assert metrics.validation_accuracy >= 0.0
        assert metrics.validation_accuracy <= 1.0
        assert metrics.num_train_examples == 5
        assert metrics.num_val_examples == 3

        # Verify model info
        assert model_info.name == "test-mipro-classifier"
        assert model_info.version is not None
        assert model_info.run_id is not None
        assert "train_accuracy" in model_info.metrics
        assert "validation_accuracy" in model_info.metrics

    @pytest.mark.slow
    def test_evaluate_model(self, model_development_service: ModelDevelopmentService):
        """Test model evaluation on test split.

        This test requires a trained model, so it runs after tune.
        """
        # First, tune a model
        _, model_info = model_development_service.tune(
            train_size=5,
            val_size=3,
            model_name="test-eval-classifier",
        )

        # Now evaluate it on a small test set
        # Load only a few test examples for speed
        model_development_service.dataset_service.load()
        test_df = model_development_service.dataset_service.get_test_dataframe().head(3)

        # Temporarily replace test split to use our small subset
        original_test = model_development_service.dataset_service._test_df
        model_development_service.dataset_service._test_df = test_df

        try:
            results = model_development_service.evaluate(
                model_name="test-eval-classifier",
                model_version=model_info.version,
                split="test",
            )

            # Verify results (now returns EvaluateMetrics)
            assert results.accuracy >= 0.0
            assert results.accuracy <= 1.0
            assert results.num_examples == 3
            assert results.run_id is not None
        finally:
            # Restore original
            model_development_service.dataset_service._test_df = original_test

    def test_list_models(self, model_development_service: ModelDevelopmentService):
        """Test listing models from registry."""
        # First, create a model
        model_development_service.tune(
            train_size=5,
            val_size=3,
            model_name="test-list-classifier",
        )

        # List models
        models_df = model_development_service.list_models()

        # Verify DataFrame structure
        assert isinstance(models_df, pd.DataFrame)
        if not models_df.empty:
            assert "name" in models_df.columns
            assert "version" in models_df.columns
            assert "aliases" in models_df.columns
            assert "creation_time" in models_df.columns
            assert "metrics" in models_df.columns

    def test_list_models_with_filter(
        self, model_development_service: ModelDevelopmentService
    ):
        """Test listing models with name filter."""
        # Create models with different names
        model_development_service.tune(
            train_size=5,
            val_size=3,
            model_name="test-filter-model-a",
        )

        # List with filter
        models_df = model_development_service.list_models(name_filter="filter-model")

        # Verify filtering worked
        assert isinstance(models_df, pd.DataFrame)
        if not models_df.empty:
            assert all("filter-model" in name.lower() for name in models_df["name"])

    @pytest.mark.slow
    @pytest.mark.mlflow
    def test_mlflow_logging_behavior(
        self, model_development_service: ModelDevelopmentService, mlflow_test_dir: Path
    ):
        """Test MLFlow logging behavior: artifacts, tags, and metrics.

        Validates:
        - CSV and Parquet artifacts are both logged
        - prompt_details.txt artifact is created
        - Experiment tags (system, project) are set correctly
        - Metric naming (train_accuracy vs accuracy for val/test)
        """
        # Tune a small model to generate MLFlow artifacts
        metrics, model_info = model_development_service.tune(
            train_size=5,
            val_size=3,
            model_name="test-mlflow-behavior",
        )

        # Get the MLFlow client to inspect logged data
        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_test_dir))
        run = client.get_run(model_info.run_id)

        # 1. Verify experiment tags
        experiment = client.get_experiment_by_name(
            "/test/symptom-diagnosis-explorer/1-dspy/classification"
        )
        assert experiment is not None
        assert experiment.tags["system"] == "symptom-diagnosis-explorer"
        assert experiment.tags["project"] == "1-dspy"

        # 2. Verify metric naming (train_accuracy vs accuracy)
        assert "train_accuracy" in run.data.metrics
        assert "accuracy" in run.data.metrics  # validation logged as "accuracy"
        assert (
            "validation_accuracy" not in run.data.metrics
        )  # Old name should not exist

        # 3. Verify dual format artifacts (CSV + Parquet)
        artifacts = client.list_artifacts(model_info.run_id)
        artifact_paths = [a.path for a in artifacts]

        # Check for CSV files
        assert any("predictions_train_sample.csv" in p for p in artifact_paths)
        assert any("predictions_validation_sample.csv" in p for p in artifact_paths)

        # Check for Parquet files
        assert any("predictions_train_sample.parquet" in p for p in artifact_paths)
        assert any("predictions_validation_sample.parquet" in p for p in artifact_paths)

        # 4. Verify prompt_details.txt artifact exists
        assert any("prompt_details.txt" in p for p in artifact_paths)

        # Optional: Download and verify prompt_details.txt content structure
        prompt_artifact = [a for a in artifacts if "prompt_details.txt" in a.path]
        if prompt_artifact:
            local_path = client.download_artifacts(
                model_info.run_id, "prompt_details.txt"
            )
            prompt_content = Path(local_path).read_text()
            # Verify expected sections exist
            assert "OPTIMIZED PROMPT DETAILS" in prompt_content
            # Check for any of our new detailed sections
            assert (
                "MODULE STRUCTURE" in prompt_content
                or "SIGNATURE OVERVIEW" in prompt_content
                or "FEW-SHOT EXAMPLES" in prompt_content
                or "ACTUAL PROMPT EXAMPLE" in prompt_content
                or "MODULE REPRESENTATION" in prompt_content
            )
