"""LangChain-based model development service.

This service implements the BaseModelService interface using LangChain's
Expression Language (LCEL) for prompt-based symptom diagnosis classification.

Unlike DSPy, LangChain does not require training/optimization. The 'tune' method
creates a chain with few-shot examples and evaluates it on training/validation sets.
"""

import json
import mlflow
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from symptom_diagnosis_explorer.models.domain import DiagnosisType
from symptom_diagnosis_explorer.models.model_development import (
    ClassificationConfig,
    EvaluateMetrics,
    FrameworkType,
    LangChainConfig,
    ModelInfo,
    TuneMetrics,
)
from symptom_diagnosis_explorer.services.ml_models.base import BaseModelService
from symptom_diagnosis_explorer.services.ml_models.registry import FrameworkRegistry


class DiagnosisOutput(BaseModel):
    """Structured output schema for diagnosis predictions."""

    diagnosis: str = Field(description="The predicted diagnosis")


def _get_prompt_template(diagnosis_types: list[str]) -> ChatPromptTemplate:
    """Get the prompt template for symptom diagnosis classification.

    Few-shot examples are managed directly within the template.
    If you need to add few-shot examples, add them directly to the
    messages list below.

    Args:
        diagnosis_types: List of valid diagnosis types (enum values).

    Returns:
        Configured ChatPromptTemplate with dynamically generated diagnosis list.
    """
    # Format diagnosis types as numbered list for better readability
    # Sort for consistency
    sorted_types = sorted(diagnosis_types)
    diagnosis_list = "\n".join(f"{i + 1}. {dt}" for i, dt in enumerate(sorted_types))

    system_prompt = f"""You are a medical diagnostic assistant specializing in symptom-to-diagnosis classification.

Your task is to analyze patient symptom descriptions and predict the most likely diagnosis from a predefined list.

VALID DIAGNOSIS CATEGORIES:
{diagnosis_list}

CRITICAL INSTRUCTIONS:
- You MUST respond with ONLY the diagnosis name from the list above
- Match the exact spelling and format (lowercase, with spaces between words)
- Do NOT include explanations, reasoning, or additional text
- Do NOT include numbers or bullet points in your response
- Examples of correct responses: "common cold", "diabetes", "urinary tract infection"

Analyze the symptoms carefully and return the single most likely diagnosis."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            # Add few-shot examples here if needed:
            # ("human", "Patient symptoms: example symptoms"),
            # ("ai", "example diagnosis"),
            ("human", "Patient symptoms: {symptoms}"),
        ]
    )


@FrameworkRegistry.register(FrameworkType.LANGCHAIN)
class LangChainModelService(BaseModelService):
    """LangChain-based model development service.

    This service uses LangChain's Expression Language (LCEL) to create
    prompt-based classification chains. It supports:
    - Zero-shot and few-shot prompting
    - Structured output for reliable parsing
    - MLFlow tracking with execution traces and token usage
    - Model versioning via config artifacts (not model serialization)
    - Evaluation on train/validation/test splits

    Since LangChain doesn't require training, the 'tune' method:
    1. Creates a chain with configured prompts/examples
    2. Evaluates on training and validation sets
    3. Logs the configuration and traces to MLFlow
    4. Returns metrics and model info
    """

    def __init__(self, config: ClassificationConfig) -> None:
        """Initialize LangChain model service.

        Args:
            config: Classification configuration with LangChain framework config.

        Raises:
            ValueError: If config.framework_config is not LangChainConfig.
        """
        super().__init__(config)

        # Validate framework config type
        if not isinstance(config.framework_config, LangChainConfig):
            raise ValueError(
                f"Expected LangChainConfig, got {type(config.framework_config).__name__}"
            )
        self.langchain_config = config.framework_config

        # Setup MLFlow experiment
        self._setup_mlflow_experiment("langchain")

        # Enable MLflow autolog for tracing
        # This enables execution traces with token usage tracking
        # Note: LangChain autolog does not serialize models - we maintain config-only approach
        mlflow.langchain.autolog(
            log_traces=True,  # Enable execution traces with token usage
            disable=False,
            silent=False,
        )

    @property
    def requires_training(self) -> bool:
        """LangChain does not require training."""
        return False

    @property
    def framework_type(self) -> str:
        """Framework identifier."""
        return FrameworkType.LANGCHAIN

    def _create_classification_chain(self):
        """Create LangChain LCEL chain for classification.

        Few-shot examples are managed directly in the prompt templates.
        Diagnosis types are taken directly from the DiagnosisType enum.

        Returns:
            LangChain runnable chain (prompt | LLM | output parser).
        """
        # Get diagnosis types from enum (guaranteed to have all valid types)
        diagnosis_types = [dt.value for dt in DiagnosisType]

        # Get prompt template with dynamically generated diagnosis list
        prompt = _get_prompt_template(diagnosis_types)

        # Create LLM
        # Extract model name from "ollama/model-name" format
        model_name = self.config.lm_config.model.replace("ollama/", "")

        llm = ChatOllama(
            model=model_name,
            temperature=self.config.lm_config.temperature,
            num_predict=self.config.lm_config.max_tokens,
        )

        # Create chain
        # Note: For Ollama models, structured output with with_structured_output()
        # can be unreliable. We'll use a simple chain and parse the output manually.
        chain = prompt | llm

        return chain

    def _evaluate_on_dataset(
        self,
        chain,
        df: pd.DataFrame,
        split: str,
    ) -> float:
        """Evaluate chain on dataset and log artifacts.

        Args:
            chain: LangChain runnable to evaluate.
            df: Dataset to evaluate on.
            split: Split name for logging (train, validation, test).

        Returns:
            Accuracy score (fraction of correct predictions).
        """
        predictions = []
        actuals = []

        for _, row in df.iterrows():
            result = chain.invoke({"symptoms": row["symptoms"]})

            # Extract diagnosis from result
            # The result should be an AIMessage with content
            if hasattr(result, "content"):
                pred_diagnosis = result.content.strip()
            else:
                pred_diagnosis = str(result).strip()

            predictions.append(pred_diagnosis)
            actuals.append(row["diagnosis"])

        # Calculate accuracy (case-insensitive, strip whitespace)
        correct = sum(
            pred.strip().lower() == actual.strip().lower()
            for pred, actual in zip(predictions, actuals)
        )
        accuracy = correct / len(df) if len(df) > 0 else 0.0

        # Log metrics
        metric_name = f"{split}_accuracy"
        mlflow.log_metric(metric_name, accuracy)

        # Log dataset info
        self._log_dataset_info(split, len(df))

        # Log sample predictions as artifact
        sample_size = min(self.config.artifact_sample_size, len(df))
        if sample_size > 0:
            sample_predictions = pd.DataFrame(
                {
                    "symptoms": [actuals[i] for i in range(sample_size)],
                    "actual": [actuals[i] for i in range(sample_size)],
                    "predicted": [predictions[i] for i in range(sample_size)],
                    "correct": [
                        predictions[i].strip().lower() == actuals[i].strip().lower()
                        for i in range(sample_size)
                    ],
                }
            )

            artifact_path = f"{split}_sample_predictions.csv"
            sample_predictions.to_csv(artifact_path, index=False)
            mlflow.log_artifact(artifact_path)
            Path(artifact_path).unlink()  # Clean up

        return accuracy

    def tune(
        self,
        train_size: int | None = None,
        val_size: int | None = None,
        model_name: str = "symptom-classifier",
    ) -> tuple[TuneMetrics, ModelInfo]:
        """Create and evaluate LangChain chain (no training required).

        For LangChain, 'tuning' means:
        1. Load training and validation datasets
        2. Extract few-shot examples from training data (if configured)
        3. Create the chain with configured prompts
        4. Evaluate on both train and validation sets
        5. Log configuration and chain to MLFlow
        6. Register the model version
        7. Return metrics and model info

        Args:
            train_size: Optional limit on training examples for few-shot and evaluation.
            val_size: Optional limit on validation examples for evaluation.
            model_name: Name for model registry.

        Returns:
            Tuple of (TuneMetrics, ModelInfo).
        """
        # Load datasets
        self.dataset_service.load()
        train_df, val_df = self.dataset_service.get_train_validation_split()

        # Apply size limits
        if train_size:
            train_df = train_df.head(train_size)
        if val_size:
            val_df = val_df.head(val_size)

        # Create chain (few-shot examples are hardcoded in the prompt template)
        chain = self._create_classification_chain()

        # Start MLFlow run
        with mlflow.start_run(run_id=self.config.mlflow_run_id) as run:
            # Log parameters
            mlflow.log_params(
                {
                    "framework": "langchain",
                    "requires_training": False,
                    "lm_model": self.config.lm_config.model,
                    "lm_temperature": self.config.lm_config.temperature,
                    "lm_max_tokens": self.config.lm_config.max_tokens,
                    "prompt_template_name": self.langchain_config.prompt_template_name,
                    "chain_type": self.langchain_config.chain_type,
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                }
            )

            # Evaluate on both splits
            train_accuracy = self._evaluate_on_dataset(chain, train_df, "train")
            val_accuracy = self._evaluate_on_dataset(chain, val_df, "validation")

            # Log configuration as artifact
            config_dict = {
                "framework": "langchain",
                "template_name": self.langchain_config.prompt_template_name,
                "chain_type": self.langchain_config.chain_type,
                "lm_config": {
                    "model": self.config.lm_config.model,
                    "temperature": self.config.lm_config.temperature,
                    "max_tokens": self.config.lm_config.max_tokens,
                },
            }

            config_path = "chain_config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            mlflow.log_artifact(config_path)
            Path(config_path).unlink()

            # Don't log the chain to MLFlow to avoid LangChain-MLFlow integration
            # We reconstruct chains from the config artifact during evaluation

            # Create registered model (if it doesn't exist) and version
            client = mlflow.tracking.MlflowClient()
            try:
                client.create_registered_model(model_name)
            except mlflow.exceptions.MlflowException:
                # Model already exists
                pass

            # Create model version pointing to this run's artifacts
            # We use the artifact URI directly since we don't have an MLmodel file
            artifact_uri = f"{run.info.artifact_uri}"
            model_version = client.create_model_version(
                name=model_name,
                source=artifact_uri,
                run_id=run.info.run_id,
            )

            # Set aliases
            for alias in self.config.model_aliases:
                client.set_registered_model_alias(
                    model_name, alias, model_version.version
                )

            # Build response
            metrics = TuneMetrics(
                train_accuracy=train_accuracy,
                validation_accuracy=val_accuracy,
                num_train_examples=len(train_df),
                num_val_examples=len(val_df),
            )

            model_info = ModelInfo(
                name=model_name,
                version=str(model_version.version),
                run_id=run.info.run_id,
                metrics={
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": val_accuracy,
                },
            )

            return metrics, model_info

    def evaluate(
        self,
        model_name: str,
        model_version: str | None = None,
        split: str = "test",
        eval_size: int | None = None,
    ) -> EvaluateMetrics:
        """Evaluate a saved LangChain model.

        This method recreates the chain from the saved configuration and
        evaluates it on the specified dataset split.

        Args:
            model_name: Name of model in MLFlow registry.
            model_version: Specific version or None for latest.
            split: Dataset split to evaluate on (train, validation, test).
            eval_size: Optional limit on evaluation examples.

        Returns:
            EvaluateMetrics with accuracy and run information.
        """
        # Load dataset split
        self.dataset_service.load()
        if split == "train":
            df = self.dataset_service.get_train_dataframe()
        elif split == "validation":
            df = self.dataset_service.get_validation_dataframe()
        elif split == "test":
            df = self.dataset_service.get_test_dataframe()
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be train, validation, or test."
            )

        if eval_size:
            df = df.head(eval_size)

        # For LangChain, we don't load anything from MLFlow at evaluation time
        # We just use the current service's configuration (prompts and settings)
        # This avoids any MLFlow dependencies during inference
        # Few-shot examples are managed directly in the prompt templates

        # Create chain using current configuration
        chain = self._create_classification_chain()

        # Evaluate
        with mlflow.start_run(run_id=self.config.mlflow_run_id) as run:
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "model_version": model_version or "latest",
                    "split": split,
                    "eval_size": len(df),
                }
            )

            accuracy = self._evaluate_on_dataset(chain, df, split)

            return EvaluateMetrics(
                accuracy=accuracy,
                num_examples=len(df),
                run_id=run.info.run_id,
            )
