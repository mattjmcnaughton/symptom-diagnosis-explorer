"""Integration tests for the PydanticAIModelService with Ollama LLMs.

This test module requires Ollama models to be available locally.
The entire module will skip if required models are not found.

Required models:
- qwen3:1.7b (primary test model)
"""

from pathlib import Path

import pytest

from tests.integration.conftest import check_all_required_models
from symptom_diagnosis_explorer.models.model_development import (
    ClassificationConfig,
    LMConfig,
    PydanticAIConfig,
)
from symptom_diagnosis_explorer.services.ml_models.pydantic_ai import (
    PydanticAIModelService,
)


# Required Ollama models for this test module
REQUIRED_MODELS = ["qwen3:1.7b"]


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
def pydantic_ai_config_zero_shot(
    test_model_name: str, mlflow_test_dir: Path
) -> ClassificationConfig:
    """Create test configuration for Pydantic-AI with zero-shot prompting.

    Args:
        test_model_name: Name of Ollama model to use.
        mlflow_test_dir: Path to temp MLFlow directory.

    Returns:
        ClassificationConfig for zero-shot testing.
    """
    return ClassificationConfig(
        lm_config=LMConfig(
            model=f"ollama/{test_model_name}",
            temperature=0.0,
            max_tokens=200,
        ),
        framework_config=PydanticAIConfig(
            output_mode="native",
            num_few_shot_examples=0,  # Zero-shot
        ),
        mlflow_experiment_name="/test/symptom-diagnosis-explorer/3-pydantic-ai/classification",
        mlflow_experiment_project="3-pydantic-ai",
        mlflow_tracking_uri=str(mlflow_test_dir),
        mlflow_run_id=None,  # Create new runs for each test
        model_aliases=["test"],
        artifact_sample_size=5,  # Small sample for tests
        dataset_identifier="gretelai/symptom_to_diagnosis",
    )


@pytest.fixture
def pydantic_ai_config_few_shot(
    test_model_name: str, mlflow_test_dir: Path
) -> ClassificationConfig:
    """Create test configuration for Pydantic-AI with few-shot prompting.

    Args:
        test_model_name: Name of Ollama model to use.
        mlflow_test_dir: Path to temp MLFlow directory.

    Returns:
        ClassificationConfig for few-shot testing.
    """
    return ClassificationConfig(
        lm_config=LMConfig(
            model=f"ollama/{test_model_name}",
            temperature=0.0,
            max_tokens=200,
        ),
        framework_config=PydanticAIConfig(
            output_mode="native",
            num_few_shot_examples=3,  # Few-shot
        ),
        mlflow_experiment_name="/test/symptom-diagnosis-explorer/3-pydantic-ai/classification-few-shot",
        mlflow_experiment_project="3-pydantic-ai",
        mlflow_tracking_uri=str(mlflow_test_dir),
        mlflow_run_id=None,
        model_aliases=["test-few-shot"],
        artifact_sample_size=5,
        dataset_identifier="gretelai/symptom_to_diagnosis",
    )


@pytest.fixture
def pydantic_ai_service_zero_shot(
    pydantic_ai_config_zero_shot: ClassificationConfig,
) -> PydanticAIModelService:
    """Create PydanticAIModelService instance with zero-shot config.

    Args:
        pydantic_ai_config_zero_shot: Zero-shot test configuration.

    Returns:
        Configured PydanticAIModelService.
    """
    return PydanticAIModelService(pydantic_ai_config_zero_shot)


@pytest.fixture
def pydantic_ai_service_few_shot(
    pydantic_ai_config_few_shot: ClassificationConfig,
) -> PydanticAIModelService:
    """Create PydanticAIModelService instance with few-shot config.

    Args:
        pydantic_ai_config_few_shot: Few-shot test configuration.

    Returns:
        Configured PydanticAIModelService.
    """
    return PydanticAIModelService(pydantic_ai_config_few_shot)


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
class TestPydanticAIModelService:
    """Integration tests for PydanticAIModelService with Ollama models."""

    @pytest.mark.slow
    def test_tune_zero_shot_with_small_dataset(
        self, pydantic_ai_service_zero_shot: PydanticAIModelService
    ):
        """Test Pydantic-AI tuning with zero-shot prompting.

        For Pydantic-AI, 'tuning' means creating an agent and evaluating it
        (no actual optimization occurs).

        This test uses minimal data (5 train, 3 validation) to verify the
        pipeline works end-to-end with Ollama.
        """
        # Tune with very small dataset for speed
        metrics, model_info = pydantic_ai_service_zero_shot.tune(
            train_size=5,
            val_size=3,
            model_name="test-pydantic-ai-zero-shot",
        )

        # Verify metrics
        assert metrics.train_accuracy >= 0.0
        assert metrics.train_accuracy <= 1.0
        assert metrics.validation_accuracy >= 0.0
        assert metrics.validation_accuracy <= 1.0
        assert metrics.num_train_examples == 5
        assert metrics.num_val_examples == 3

        # Verify model info
        assert model_info.name == "test-pydantic-ai-zero-shot"
        assert model_info.version is not None
        assert model_info.run_id is not None
        assert "train_accuracy" in model_info.metrics
        assert "validation_accuracy" in model_info.metrics

    @pytest.mark.slow
    def test_tune_few_shot_with_small_dataset(
        self, pydantic_ai_service_few_shot: PydanticAIModelService
    ):
        """Test Pydantic-AI tuning with few-shot prompting.

        This test verifies that few-shot configuration is correctly handled
        in the agent creation process.
        """
        # Tune with very small dataset for speed
        # Note: Using 10 training examples to ensure we have enough for few-shot configuration
        metrics, model_info = pydantic_ai_service_few_shot.tune(
            train_size=10,
            val_size=3,
            model_name="test-pydantic-ai-few-shot",
        )

        # Verify metrics
        assert metrics.train_accuracy >= 0.0
        assert metrics.train_accuracy <= 1.0
        assert metrics.validation_accuracy >= 0.0
        assert metrics.validation_accuracy <= 1.0
        assert metrics.num_train_examples == 10
        assert metrics.num_val_examples == 3

        # Verify model info
        assert model_info.name == "test-pydantic-ai-few-shot"
        assert model_info.version is not None
        assert model_info.run_id is not None

    @pytest.mark.slow
    def test_evaluate_on_test_split(
        self, pydantic_ai_service_zero_shot: PydanticAIModelService
    ):
        """Test model evaluation on test split.

        This test:
        1. Tunes an agent (creates agent + evaluates on train/val)
        2. Evaluates the same agent configuration on test split
        """
        # First tune to create a model
        _, model_info = pydantic_ai_service_zero_shot.tune(
            train_size=5,
            val_size=3,
            model_name="test-pydantic-ai-eval",
        )

        # Now evaluate on test split
        eval_metrics = pydantic_ai_service_zero_shot.evaluate(
            model_name="test-pydantic-ai-eval",
            model_version=model_info.version,
            split="test",
            eval_size=5,  # Small test set for speed
        )

        # Verify evaluation metrics
        assert eval_metrics.accuracy >= 0.0
        assert eval_metrics.accuracy <= 1.0
        assert eval_metrics.num_examples == 5
        assert eval_metrics.run_id is not None

    @pytest.mark.slow
    def test_evaluate_latest_version(
        self, pydantic_ai_service_zero_shot: PydanticAIModelService
    ):
        """Test evaluation using 'latest' version.

        This verifies that we can evaluate models without specifying version.
        """
        # First tune to create a model
        pydantic_ai_service_zero_shot.tune(
            train_size=5,
            val_size=3,
            model_name="test-pydantic-ai-latest",
        )

        # Evaluate using latest version
        eval_metrics = pydantic_ai_service_zero_shot.evaluate(
            model_name="test-pydantic-ai-latest",
            model_version=None,  # Use latest
            split="test",
            eval_size=5,
        )

        # Verify evaluation metrics
        assert eval_metrics.accuracy >= 0.0
        assert eval_metrics.accuracy <= 1.0
        assert eval_metrics.num_examples == 5

    def test_requires_training_property(
        self, pydantic_ai_service_zero_shot: PydanticAIModelService
    ):
        """Test that requires_training returns False for Pydantic-AI."""
        assert pydantic_ai_service_zero_shot.requires_training is False

    def test_framework_type_property(
        self, pydantic_ai_service_zero_shot: PydanticAIModelService
    ):
        """Test that framework_type returns correct identifier."""
        assert pydantic_ai_service_zero_shot.framework_type == "pydantic-ai"

    def test_invalid_config_type_raises_error(
        self, test_model_name: str, mlflow_test_dir: Path
    ):
        """Test that PydanticAIModelService raises error with DSPy config."""
        from symptom_diagnosis_explorer.models.model_development import (
            DSPyConfig,
            OptimizerConfig,
        )

        # Create config with DSPyConfig (wrong type for Pydantic-AI)
        dspy_config = ClassificationConfig(
            lm_config=LMConfig(
                model=f"ollama/{test_model_name}",
                temperature=0.0,
                max_tokens=200,
            ),
            framework_config=DSPyConfig(optimizer_config=OptimizerConfig()),
            mlflow_experiment_name="/test/invalid",
            mlflow_experiment_project="test",
            mlflow_tracking_uri=str(mlflow_test_dir),
        )

        with pytest.raises(ValueError, match="Expected PydanticAIConfig"):
            PydanticAIModelService(dspy_config)
