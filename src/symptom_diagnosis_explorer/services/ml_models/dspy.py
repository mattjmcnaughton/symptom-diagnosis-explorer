"""DSPy-based model development service with MLFlow tracking."""

import os
from typing import Any

import dspy
import mlflow
import mlflow.data
import mlflow.dspy
import pandas as pd
from dspy.teleprompt import BootstrapFewShot, MIPROv2, Teleprompter

from symptom_diagnosis_explorer.models.domain import SymptomDiagnosisSignature
from symptom_diagnosis_explorer.models.model_development import (
    ClassificationConfig,
    DSPyConfig,
    EvaluateMetrics,
    FrameworkType,
    ModelInfo,
    OptimizerType,
    TuneMetrics,
)
from symptom_diagnosis_explorer.services.ml_models.base import BaseModelService
from symptom_diagnosis_explorer.services.ml_models.registry import FrameworkRegistry


@FrameworkRegistry.register(FrameworkType.DSPY)
class DSPyModelService(BaseModelService):
    """DSPy-based model development service.

    This service implements the BaseModelService interface for DSPy framework,
    handling model compilation/optimization, evaluation, and MLFlow tracking.

    DSPy requires training through compilation with optimizers like
    BootstrapFewShot or MIPROv2.
    """

    def __init__(self, config: ClassificationConfig) -> None:
        """Initialize the DSPy model service.

        Sets up DSPy language model configuration and MLFlow tracking.

        Args:
            config: Classification configuration including LM, optimizer, and MLFlow settings.
        """
        super().__init__(config)

        # Extract DSPy config from framework_config
        if not isinstance(config.framework_config, DSPyConfig):
            raise ValueError(
                f"Expected DSPyConfig, got {type(config.framework_config).__name__}"
            )
        self.dspy_config = config.framework_config

        # Configure DSPy with language model
        lm = dspy.LM(
            model=config.lm_config.model,
            temperature=config.lm_config.temperature,
            max_tokens=config.lm_config.max_tokens,
        )
        dspy.configure(lm=lm)

        # Setup MLFlow autologging for DSPy
        mlflow.dspy.autolog(
            log_compiles=False,
            log_evals=False,
            log_traces_from_compile=False,
        )

        # Setup MLFlow experiment with framework tagging
        self._setup_mlflow_experiment("dspy")

    @property
    def requires_training(self) -> bool:
        """DSPy always requires training through compilation."""
        return True

    @property
    def framework_type(self) -> str:
        """Framework type identifier."""
        return FrameworkType.DSPY

    def _convert_to_dspy_examples(self, df: pd.DataFrame) -> list[dspy.Example]:
        """Convert pandas DataFrame to list of DSPy Examples.

        Args:
            df: DataFrame with 'symptoms' and 'diagnosis' columns.

        Returns:
            List of DSPy Example objects with labels for training/evaluation.
        """
        examples = []
        for _, row in df.iterrows():
            example = dspy.Example(
                symptoms=row["symptoms"],
                diagnosis=row["diagnosis"],
            ).with_inputs("symptoms")
            examples.append(example)
        return examples

    def _create_classification_module(self) -> dspy.Module:
        """Create DSPy classification module.

        We use ChainOfThought because it prompts the LM to generate intermediate
        reasoning steps before producing the final diagnosis. This improves accuracy
        for complex medical reasoning tasks by making the model's decision process
        explicit and verifiable.

        Returns:
            DSPy module for symptom classification (ChainOfThought with signature).
        """
        return dspy.ChainOfThought(SymptomDiagnosisSignature)

    def _create_optimizer(self) -> Teleprompter:
        """Create DSPy optimizer/teleprompter based on configuration.

        Returns:
            Configured optimizer (BootstrapFewShot or MIPROv2).
        """
        optimizer_config = self.dspy_config.optimizer_config

        if optimizer_config.optimizer_type == OptimizerType.BOOTSTRAP_FEW_SHOT:
            return BootstrapFewShot(
                metric=self._classification_metric,
                max_bootstrapped_demos=optimizer_config.bootstrap_max_bootstrapped_demos,
                max_labeled_demos=optimizer_config.bootstrap_max_labeled_demos,
            )
        elif optimizer_config.optimizer_type == OptimizerType.MIPRO_V2:
            # Use prompt_model if specified, otherwise use task model
            prompt_model_identifier = (
                self.config.lm_config.prompt_model or self.config.lm_config.model
            )
            # Create dspy.LM objects for MIPROv2 (it expects LM objects, not strings)
            # Prompt model uses higher temperature for more creative instruction generation
            prompt_lm = dspy.LM(
                model=prompt_model_identifier,
                temperature=1.0,
                max_tokens=2048,
            )
            task_lm = dspy.LM(
                model=self.config.lm_config.model,
                temperature=self.config.lm_config.temperature,
                max_tokens=self.config.lm_config.max_tokens,
            )
            return MIPROv2(
                metric=self._classification_metric,
                num_threads=optimizer_config.num_threads,
                prompt_model=prompt_lm,
                task_model=task_lm,
                auto=optimizer_config.mipro_auto,
            )
        else:
            raise ValueError(
                f"Unknown optimizer type: {optimizer_config.optimizer_type}"
            )

    def _classification_metric(
        self, example: dspy.Example, prediction: dspy.Prediction, trace: Any = None
    ) -> float:
        """Metric function for evaluating classification predictions.

        Args:
            example: DSPy Example with ground truth.
            prediction: DSPy Prediction from model.
            trace: Optional trace information (unused).

        Returns:
            1.0 if prediction matches ground truth, 0.0 otherwise.
        """
        # Handle both string and DiagnosisType enum comparisons
        predicted = str(prediction.diagnosis).strip().lower()
        actual = str(example.diagnosis).strip().lower()
        return 1.0 if predicted == actual else 0.0

    def _log_predictions_artifacts(
        self,
        examples: list[dspy.Example],
        predictions: list[dspy.Prediction],
        split: str,
    ) -> None:
        """Log sample predictions as CSV and Parquet artifacts.

        Args:
            examples: List of input examples.
            predictions: List of model predictions.
            split: Dataset split name for artifact naming.
        """
        # Limit to configured sample size
        sample_size = min(self.config.artifact_sample_size, len(examples))

        # Build DataFrame with samples
        rows = []
        for i in range(sample_size):
            rows.append(
                {
                    "symptoms": examples[i].symptoms,
                    "actual_diagnosis": str(examples[i].diagnosis),
                    "predicted_diagnosis": str(predictions[i].diagnosis),
                    "correct": str(predictions[i].diagnosis).strip().lower()
                    == str(examples[i].diagnosis).strip().lower(),
                }
            )

        df = pd.DataFrame(rows)

        # Log as CSV artifact
        csv_artifact_path = f"predictions_{split}_sample.csv"
        df.to_csv(csv_artifact_path, index=False)
        mlflow.log_artifact(csv_artifact_path)
        os.remove(csv_artifact_path)

        # Log as parquet artifact
        parquet_artifact_path = f"predictions_{split}_sample.parquet"
        df.to_parquet(parquet_artifact_path, index=False)
        mlflow.log_artifact(parquet_artifact_path)
        os.remove(parquet_artifact_path)

    def _log_disagreements_artifacts(
        self,
        examples: list[dspy.Example],
        predictions: list[dspy.Prediction],
        split: str,
    ) -> None:
        """Log all prediction disagreements as CSV and Parquet artifacts.

        Args:
            examples: List of input examples.
            predictions: List of model predictions.
            split: Dataset split name for artifact naming.
        """
        # Find all disagreements
        rows = []
        for ex, pred in zip(examples, predictions):
            predicted = str(pred.diagnosis).strip().lower()
            actual = str(ex.diagnosis).strip().lower()
            if predicted != actual:
                rows.append(
                    {
                        "symptoms": ex.symptoms,
                        "actual_diagnosis": str(ex.diagnosis),
                        "predicted_diagnosis": str(pred.diagnosis),
                    }
                )

        # Only log if there are disagreements
        if rows:
            df = pd.DataFrame(rows)

            # Log as CSV artifact
            csv_artifact_path = f"disagreements_{split}.csv"
            df.to_csv(csv_artifact_path, index=False)
            mlflow.log_artifact(csv_artifact_path)
            os.remove(csv_artifact_path)

            # Log as parquet artifact
            parquet_artifact_path = f"disagreements_{split}.parquet"
            df.to_parquet(parquet_artifact_path, index=False)
            mlflow.log_artifact(parquet_artifact_path)
            os.remove(parquet_artifact_path)

            # Log disagreement count as metric
            mlflow.log_metric(f"disagreements_{split}_count", len(rows))

    def _extract_module_structure(self, compiled_module: dspy.Module) -> list[str]:
        """Extract module structure showing named predictors hierarchy.

        Args:
            compiled_module: Compiled DSPy module.

        Returns:
            List of text lines describing the module structure.
        """
        lines = []
        try:
            predictors = list(compiled_module.named_predictors())
            if predictors:
                lines.append("## MODULE STRUCTURE")
                lines.append(f"Total predictors: {len(predictors)}")
                lines.append("")
                for name, predictor in predictors:
                    lines.append(f"### Predictor: {name}")
                    lines.append(f"Type: {type(predictor).__name__}")
                    if hasattr(predictor, "signature"):
                        lines.append(
                            f"Signature: {predictor.signature.__class__.__name__}"
                        )
                    if hasattr(predictor, "demos") and predictor.demos:
                        lines.append(f"Demos: {len(predictor.demos)} examples")
                    lines.append("")
        except Exception as e:
            lines.append(f"## MODULE STRUCTURE (Error: {e})")
            lines.append("")
        return lines

    def _extract_signature_fields(self, compiled_module: dspy.Module) -> list[str]:
        """Extract detailed field metadata from signature.

        Args:
            compiled_module: Compiled DSPy module.

        Returns:
            List of text lines describing signature fields.
        """
        lines = []
        if not hasattr(compiled_module, "signature"):
            return lines

        signature = compiled_module.signature
        lines.append("## SIGNATURE OVERVIEW")
        lines.append(f"Signature class: {signature.__class__.__name__}")
        lines.append("")

        # Extract input fields with metadata
        if hasattr(signature, "input_fields"):
            lines.append("### Input Fields")
            for field_name, field in signature.input_fields.items():
                lines.append(f"**{field_name}**")
                if hasattr(field, "json_schema_extra"):
                    schema = field.json_schema_extra
                    if schema and isinstance(schema, dict):
                        if "desc" in schema:
                            lines.append(f"  Description: {schema['desc']}")
                        if "prefix" in schema:
                            lines.append(f"  Prefix: {schema['prefix']}")
                if hasattr(field, "annotation"):
                    lines.append(f"  Type: {field.annotation}")
                lines.append("")

        # Extract output fields with metadata
        if hasattr(signature, "output_fields"):
            lines.append("### Output Fields")
            for field_name, field in signature.output_fields.items():
                lines.append(f"**{field_name}**")
                if hasattr(field, "json_schema_extra"):
                    schema = field.json_schema_extra
                    if schema and isinstance(schema, dict):
                        if "desc" in schema:
                            lines.append(f"  Description: {schema['desc']}")
                        if "prefix" in schema:
                            lines.append(f"  Prefix: {schema['prefix']}")
                if hasattr(field, "annotation"):
                    lines.append(f"  Type: {field.annotation}")
                lines.append("")

        # Extract signature description
        if hasattr(signature, "__doc__") and signature.__doc__:
            lines.append("## SIGNATURE DESCRIPTION")
            lines.append(signature.__doc__.strip())
            lines.append("")

        return lines

    def _extract_instructions(self, compiled_module: dspy.Module) -> list[str]:
        """Extract optimized instructions from signature.

        Args:
            compiled_module: Compiled DSPy module.

        Returns:
            List of text lines describing instructions.
        """
        lines = []
        if not hasattr(compiled_module, "signature"):
            return lines

        signature = compiled_module.signature

        # Extract signature-level instructions if available
        if hasattr(signature, "instructions") and signature.instructions:
            lines.append("## SIGNATURE-LEVEL INSTRUCTIONS")
            lines.append(signature.instructions)
            lines.append("")

        # Extract field-level instructions
        has_field_instructions = False
        field_instructions = []
        for field_name in dir(signature):
            field = getattr(signature, field_name)
            if hasattr(field, "instructions") and field.instructions:
                if not has_field_instructions:
                    has_field_instructions = True
                field_instructions.append(f"### Field: {field_name}")
                field_instructions.append(field.instructions)
                field_instructions.append("")

        if has_field_instructions:
            lines.append("## FIELD-LEVEL INSTRUCTIONS")
            lines.extend(field_instructions)

        return lines

    def _extract_few_shot_examples(self, compiled_module: dspy.Module) -> list[str]:
        """Extract few-shot demonstration examples.

        Args:
            compiled_module: Compiled DSPy module.

        Returns:
            List of text lines describing few-shot examples.
        """
        lines = []
        if not (hasattr(compiled_module, "demos") and compiled_module.demos):
            return lines

        lines.append("## FEW-SHOT EXAMPLES")
        lines.append(f"Total examples: {len(compiled_module.demos)}")
        lines.append("")

        # Show first 5 examples in detail
        for idx, demo in enumerate(compiled_module.demos[:5], 1):
            lines.append(f"### Example {idx}")
            # Show all available attributes
            for attr in dir(demo):
                if not attr.startswith("_") and hasattr(demo, attr):
                    value = getattr(demo, attr)
                    # Skip methods and special attributes
                    if not callable(value) and attr not in [
                        "toDict",
                        "with_inputs",
                        "inputs",
                        "labels",
                    ]:
                        lines.append(f"{attr}: {value}")
            lines.append("")

        if len(compiled_module.demos) > 5:
            lines.append(f"... and {len(compiled_module.demos) - 5} more examples")
            lines.append("")

        return lines

    def _extract_lm_call_history(
        self, compiled_module: dspy.Module, train_examples: list[dspy.Example]
    ) -> list[str]:
        """Capture actual LM prompts via sample forward pass.

        Args:
            compiled_module: Compiled DSPy module.
            train_examples: Training examples for sample forward pass.

        Returns:
            List of text lines describing actual LM call.
        """
        lines = []
        if not train_examples:
            return lines

        try:
            lines.append("## ACTUAL PROMPT EXAMPLE")
            lines.append("Sample forward pass to capture actual LM prompts")
            lines.append("")

            # Get the current LM
            lm = dspy.settings.lm
            initial_history_len = len(lm.history) if hasattr(lm, "history") else 0

            # Run sample forward pass
            sample_input = train_examples[0].symptoms
            _ = compiled_module(symptoms=sample_input)

            # Extract the new history entry
            if hasattr(lm, "history") and len(lm.history) > initial_history_len:
                last_call = lm.history[-1]

                lines.append("### Sample Input")
                lines.append(f"{sample_input}")
                lines.append("")

                # Extract prompt messages
                if "messages" in last_call or "prompt" in last_call:
                    lines.append("### Messages Sent to LM")
                    messages = last_call.get("messages", last_call.get("prompt", []))
                    if isinstance(messages, list):
                        for idx, msg in enumerate(messages, 1):
                            if isinstance(msg, dict):
                                role = msg.get("role", "unknown")
                                content = msg.get("content", str(msg))
                                lines.append(f"**Message {idx} ({role}):**")
                                lines.append(content)
                                lines.append("")
                            else:
                                lines.append(f"**Message {idx}:**")
                                lines.append(str(msg))
                                lines.append("")
                    else:
                        lines.append(str(messages))
                        lines.append("")

                # Extract response
                if "response" in last_call or "outputs" in last_call:
                    lines.append("### LM Response")
                    response = last_call.get("response", last_call.get("outputs"))
                    lines.append(str(response))
                    lines.append("")

                # Extract token usage
                if "usage" in last_call:
                    lines.append("### Token Usage")
                    usage = last_call["usage"]
                    if isinstance(usage, dict):
                        for key, value in usage.items():
                            lines.append(f"{key}: {value}")
                    else:
                        lines.append(str(usage))
                    lines.append("")

                # Extract metadata
                metadata_keys = ["model", "temperature", "max_tokens", "kwargs"]
                metadata_found = False
                for key in metadata_keys:
                    if key in last_call:
                        if not metadata_found:
                            lines.append("### LM Configuration")
                            metadata_found = True
                        lines.append(f"{key}: {last_call[key]}")
                if metadata_found:
                    lines.append("")

        except Exception as e:
            lines.append(f"Error capturing LM history: {e}")
            lines.append("")

        return lines

    def _extract_and_log_prompt(
        self, compiled_module: dspy.Module, train_examples: list[dspy.Example]
    ) -> None:
        """Extract and log comprehensive prompt details from compiled module.

        Orchestrates multiple extraction helpers to capture all available prompt
        information including module structure, signatures, instructions, examples,
        and actual LM call history.

        Args:
            compiled_module: Compiled DSPy module to extract prompt from.
            train_examples: Training examples for sample forward pass.
        """
        prompt_text_lines = []

        # Add header
        prompt_text_lines.append("=" * 80)
        prompt_text_lines.append("OPTIMIZED PROMPT DETAILS")
        prompt_text_lines.append("=" * 80)
        prompt_text_lines.append("")

        # Extract all components using helper functions
        prompt_text_lines.extend(self._extract_module_structure(compiled_module))
        prompt_text_lines.extend(self._extract_signature_fields(compiled_module))
        prompt_text_lines.extend(self._extract_instructions(compiled_module))
        prompt_text_lines.extend(self._extract_few_shot_examples(compiled_module))
        prompt_text_lines.extend(
            self._extract_lm_call_history(compiled_module, train_examples)
        )

        # Fallback: if no specific data was extracted, add module representation
        if len(prompt_text_lines) <= 6:  # Only header
            prompt_text_lines.append("## MODULE REPRESENTATION")
            prompt_text_lines.append(str(compiled_module))
            prompt_text_lines.append("")

        # Save as text artifact
        prompt_text = "\n".join(prompt_text_lines)
        artifact_path = "prompt_details.txt"
        with open(artifact_path, "w") as f:
            f.write(prompt_text)
        mlflow.log_artifact(artifact_path)
        os.remove(artifact_path)

    def _evaluate_on_dataset(
        self,
        module: dspy.Module,
        examples: list[dspy.Example],
        split: str,
    ) -> float:
        """Evaluate model on dataset and log all metrics and artifacts.

        This unified evaluation function handles:
        - Running predictions once
        - Calculating accuracy from those predictions
        - Logging accuracy metrics
        - Logging dataset info
        - Logging prediction samples as parquet
        - Logging disagreements as parquet

        Args:
            module: DSPy module to evaluate.
            examples: List of examples to evaluate on.
            split: Dataset split name (train, validation, test) for logging.

        Returns:
            Accuracy score (0.0-1.0).
        """
        # Generate predictions once (don't use Evaluate to avoid double prediction pass)
        predictions = [module(symptoms=ex.symptoms) for ex in examples]

        # Calculate accuracy manually using our metric
        correct = sum(
            self._classification_metric(ex, pred)
            for ex, pred in zip(examples, predictions)
        )
        accuracy = correct / len(examples)

        # Log accuracy metric (use just "accuracy" for validation/test, "train_accuracy" for train)
        metric_name = "train_accuracy" if split == "train" else "accuracy"
        mlflow.log_metric(metric_name, accuracy)

        # Log dataset info
        self._log_dataset_info(split, len(examples))

        # Log prediction samples and disagreements as parquet
        self._log_predictions_artifacts(examples, predictions, split)
        self._log_disagreements_artifacts(examples, predictions, split)

        return accuracy

    def tune(
        self,
        train_size: int | None = None,
        val_size: int | None = None,
        model_name: str = "symptom-classifier",
    ) -> tuple[TuneMetrics, ModelInfo]:
        """Tune/optimize classification model with automatic validation evaluation.

        Loads training and validation data, optimizes the model using configured optimizer,
        evaluates on validation set, and registers the model in MLFlow.

        Args:
            train_size: Optional limit on number of training examples to use.
            val_size: Optional limit on number of validation examples to use.
            model_name: Name to register model under in MLFlow registry.

        Returns:
            Tuple of (TuneMetrics, ModelInfo) with training results and model metadata.

        Raises:
            RuntimeError: If dataset hasn't been loaded.
        """
        # Load dataset
        self.dataset_service.load()
        train_df, val_df = self.dataset_service.get_train_validation_split()

        # Limit dataset sizes if specified
        if train_size is not None:
            train_df = train_df.head(train_size)
        if val_size is not None:
            val_df = val_df.head(val_size)

        # Convert to DSPy examples
        train_examples = self._convert_to_dspy_examples(train_df)
        val_examples = self._convert_to_dspy_examples(val_df)

        # Create module and optimizer
        module = self._create_classification_module()
        optimizer = self._create_optimizer()

        # Start MLFlow run and compile/optimize
        with mlflow.start_run(run_id=self.config.mlflow_run_id) as run:
            # Log common parameters
            params_to_log = {
                "framework": "dspy",
                "requires_training": True,
                "optimizer_type": self.dspy_config.optimizer_config.optimizer_type.value,
                "lm_model": self.config.lm_config.model,
                "train_size": len(train_examples),
                "val_size": len(val_examples),
                "num_threads": self.dspy_config.optimizer_config.num_threads,
            }

            # Log optimizer-specific parameters
            if (
                self.dspy_config.optimizer_config.optimizer_type
                == OptimizerType.BOOTSTRAP_FEW_SHOT
            ):
                params_to_log.update(
                    {
                        "bootstrap_max_bootstrapped_demos": self.dspy_config.optimizer_config.bootstrap_max_bootstrapped_demos,
                        "bootstrap_max_labeled_demos": self.dspy_config.optimizer_config.bootstrap_max_labeled_demos,
                    }
                )
            elif (
                self.dspy_config.optimizer_config.optimizer_type
                == OptimizerType.MIPRO_V2
            ):
                params_to_log.update(
                    {
                        "mipro_auto": str(self.dspy_config.optimizer_config.mipro_auto),
                        "mipro_minibatch_size": self.dspy_config.optimizer_config.mipro_minibatch_size,
                        "mipro_minibatch_full_eval_steps": self.dspy_config.optimizer_config.mipro_minibatch_full_eval_steps,
                        "mipro_program_aware_proposer": self.dspy_config.optimizer_config.mipro_program_aware_proposer,
                        "mipro_data_aware_proposer": self.dspy_config.optimizer_config.mipro_data_aware_proposer,
                        "mipro_tip_aware_proposer": self.dspy_config.optimizer_config.mipro_tip_aware_proposer,
                        "mipro_fewshot_aware_proposer": self.dspy_config.optimizer_config.mipro_fewshot_aware_proposer,
                    }
                )

            mlflow.log_params(params_to_log)

            # Compile with optimizer-specific arguments
            compile_kwargs = {
                "student": module,
                "trainset": train_examples,
            }

            # Add MIPROv2-specific compile arguments
            if (
                self.dspy_config.optimizer_config.optimizer_type
                == OptimizerType.MIPRO_V2
            ):
                compile_kwargs.update(
                    {
                        "valset": val_examples,
                        "minibatch": True,
                        "minibatch_size": self.dspy_config.optimizer_config.mipro_minibatch_size,
                        "minibatch_full_eval_steps": self.dspy_config.optimizer_config.mipro_minibatch_full_eval_steps,
                        "program_aware_proposer": self.dspy_config.optimizer_config.mipro_program_aware_proposer,
                        "data_aware_proposer": self.dspy_config.optimizer_config.mipro_data_aware_proposer,
                        "tip_aware_proposer": self.dspy_config.optimizer_config.mipro_tip_aware_proposer,
                        "fewshot_aware_proposer": self.dspy_config.optimizer_config.mipro_fewshot_aware_proposer,
                    }
                )

            compiled_module = optimizer.compile(**compile_kwargs)

            # Extract and log the optimized prompt details
            self._extract_and_log_prompt(compiled_module, train_examples)

            # Evaluate on both train and validation sets using unified helper
            train_accuracy = self._evaluate_on_dataset(
                compiled_module, train_examples, "train"
            )
            validation_accuracy = self._evaluate_on_dataset(
                compiled_module, val_examples, "validation"
            )

            # Manually log the DSPy program (autolog doesn't always work correctly)
            mlflow.dspy.log_model(compiled_module, "model")

            # Register model in MLFlow registry
            model_uri = f"runs:/{run.info.run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)

            # Set configured model aliases
            client = mlflow.tracking.MlflowClient()
            for alias in self.config.model_aliases:
                client.set_registered_model_alias(
                    model_name,
                    alias,
                    model_version.version,
                )

            # Build metrics response
            metrics = TuneMetrics(
                train_accuracy=train_accuracy,
                validation_accuracy=validation_accuracy,
                num_train_examples=len(train_examples),
                num_val_examples=len(val_examples),
            )

            # Build model info response
            model_info = ModelInfo(
                name=model_name,
                version=str(model_version.version),
                run_id=run.info.run_id,
                metrics={
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": validation_accuracy,
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
        """Evaluate a saved model on specified dataset split.

        Args:
            model_name: Name of model in MLFlow registry.
            model_version: Specific version to load (if None, loads from 'latest' alias).
            split: Dataset split to evaluate on ("train", "validation", or "test").
            eval_size: Optional limit on number of evaluation examples to use.

        Returns:
            EvaluateMetrics with accuracy, example count, and run ID.

        Raises:
            ValueError: If split is invalid.
            RuntimeError: If dataset hasn't been loaded.
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

        # Limit dataset size if specified
        if eval_size is not None:
            df = df.head(eval_size)

        # Convert to DSPy examples
        examples = self._convert_to_dspy_examples(df)

        # Load model from registry
        if model_version:
            model_uri = f"models:/{model_name}/{model_version}"
        else:
            model_uri = f"models:/{model_name}/latest"
        loaded_model = mlflow.dspy.load_model(model_uri)

        # Evaluate with MLFlow run tracking
        with mlflow.start_run(run_id=self.config.mlflow_run_id) as run:
            mlflow.log_params(
                {
                    "framework": "dspy",
                    "model_name": model_name,
                    "model_version": model_version or "latest",
                    "split": split,
                }
            )

            # Use unified evaluation helper
            accuracy = self._evaluate_on_dataset(loaded_model, examples, split)

            # Build and return metrics response
            return EvaluateMetrics(
                accuracy=accuracy,
                num_examples=len(examples),
                run_id=run.info.run_id,
            )
