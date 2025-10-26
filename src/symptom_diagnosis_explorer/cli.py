"""Command-line interface for the Symptom Diagnosis Explorer."""

import typer
from rich.console import Console
from rich.table import Table
from typing import Literal, Optional
from typing_extensions import Annotated

from symptom_diagnosis_explorer.commands import (
    DatasetListCommand,
    DatasetListRequest,
    DatasetSummaryCommand,
    DatasetSummaryRequest,
    EvaluateCommand,
    EvaluateRequest,
    ListModelsCommand,
    ListModelsRequest,
    TuneCommand,
    TuneRequest,
)
from symptom_diagnosis_explorer.models.model_development import OptimizerType

app = typer.Typer(
    name="symptom-diagnosis-explorer",
    help="Explore and analyze symptom diagnosis using ML models.",
    add_completion=False,
)

dataset_app = typer.Typer(
    name="dataset",
    help="Dataset management commands.",
)
app.add_typer(dataset_app, name="dataset")

classify_app = typer.Typer(
    name="classify",
    help="Symptom classification with DSPy.",
)
app.add_typer(classify_app, name="classify")

console = Console()


@dataset_app.command("list")
def dataset_list(
    split: Annotated[
        str,
        typer.Option(
            help="Dataset split to display (train/test/all)",
        ),
    ] = "all",
    rows: Annotated[
        int,
        typer.Option(
            help="Number of rows to display",
        ),
    ] = 5,
) -> None:
    """Display the first N rows of the dataset."""
    console.print(f"[bold]Loading {split} dataset...[/bold]")

    try:
        # Create request and execute command
        request = DatasetListRequest(split=split, rows=rows)
        command = DatasetListCommand()
        response = command.execute(request)

        # Display results
        table = Table(title=f"{split.capitalize()} Dataset - First {rows} Rows")

        # Add columns
        for column in response.df.columns:
            table.add_column(column, style="cyan", no_wrap=False)

        # Add rows
        for _, row in response.df.iterrows():
            table.add_row(*[str(val) for val in row])

        console.print(table)
        console.print(
            f"\n[green]Total rows in {split} dataset: {response.total_rows}[/green]"
        )

    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        raise typer.Exit(1)


@dataset_app.command("summary")
def dataset_summary(
    split: Annotated[
        str,
        typer.Option(
            help="Dataset split to summarize (train/test/all)",
        ),
    ] = "all",
) -> None:
    """Display summary statistics for the dataset."""
    console.print(f"[bold]Loading {split} dataset...[/bold]")

    try:
        # Create request and execute command
        request = DatasetSummaryRequest(split=split)
        command = DatasetSummaryCommand()
        response = command.execute(request)

        # Display basic info
        console.print(f"\n[bold cyan]{split.capitalize()} Dataset Summary[/bold cyan]")
        console.print(f"Total rows: {response.stats['total_rows']}")
        console.print(f"Total columns: {response.stats['total_columns']}")
        console.print(f"Columns: {', '.join(response.stats['columns'])}\n")

        # Get descriptive statistics for numeric columns
        numeric_stats = response.stats["numeric_stats"]

        if not numeric_stats.empty:
            # Create table for numeric statistics
            stats_table = Table(title="Numeric Column Statistics")
            stats_table.add_column("Statistic", style="bold magenta")

            for column in numeric_stats.columns:
                stats_table.add_column(column, style="cyan")

            # Add rows for each statistic
            for stat_name in numeric_stats.index:
                row_data = [stat_name]
                for column in numeric_stats.columns:
                    value = numeric_stats.loc[stat_name, column]
                    row_data.append(
                        f"{value:.2f}" if isinstance(value, float) else str(value)
                    )
                stats_table.add_row(*row_data)

            console.print(stats_table)
        else:
            console.print("[yellow]No numeric columns found in the dataset.[/yellow]")

        # Display info about non-numeric columns
        non_numeric_info = response.stats["non_numeric_info"]
        if len(non_numeric_info) > 0:
            console.print("\n[bold]Non-numeric columns:[/bold]")
            for col, info in non_numeric_info.items():
                console.print(
                    f"  • {col}: {info['unique_count']} unique values, {info['null_count']} null values"
                )

    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        raise typer.Exit(1)


@classify_app.command("tune")
def classify_tune(
    optimizer: Annotated[
        OptimizerType,
        typer.Option(
            help="Optimizer type (bootstrap or mipro)",
        ),
    ] = OptimizerType.BOOTSTRAP_FEW_SHOT,
    train_size: Annotated[
        Optional[int],
        typer.Option(
            help="Limit training examples (None = use all)",
        ),
    ] = None,
    val_size: Annotated[
        Optional[int],
        typer.Option(
            help="Limit validation examples (None = use all)",
        ),
    ] = None,
    model_name: Annotated[
        str,
        typer.Option(
            help="Model name for MLFlow registry",
        ),
    ] = "symptom-classifier",
    project: Annotated[
        str,
        typer.Option(
            help="Project identifier for MLFlow (e.g., 1-dspy)",
        ),
    ] = "default",
    experiment_name: Annotated[
        str,
        typer.Option(
            help="Experiment name for MLFlow (auto-constructs /symptom-diagnosis-explorer/{project}/{experiment-name})",
        ),
    ] = "tune",
    lm_model: Annotated[
        str,
        typer.Option(
            help="Language model identifier",
        ),
    ] = "ollama/qwen3:8b",
    num_threads: Annotated[
        int,
        typer.Option(
            help="Number of parallel threads for optimization",
        ),
    ] = 4,
    mlflow_tracking_uri: Annotated[
        str,
        typer.Option(
            help="MLFlow tracking server URI",
        ),
    ] = "http://localhost:5001",
    # Bootstrap-specific options
    bootstrap_max_bootstrapped_demos: Annotated[
        int,
        typer.Option(
            help="Bootstrap: Maximum number of bootstrapped demonstrations",
        ),
    ] = 3,
    bootstrap_max_labeled_demos: Annotated[
        int,
        typer.Option(
            help="Bootstrap: Maximum number of labeled demonstrations",
        ),
    ] = 4,
    # MIPROv2-specific options
    mipro_auto: Annotated[
        str,
        typer.Option(
            help="MiPro: Auto mode - 'light' (fast, 6 candidates), 'medium' (balanced, 12 candidates), 'heavy' (thorough, 18 candidates), 'none' (manual)",
        ),
    ] = "light",
    mipro_minibatch_size: Annotated[
        int,
        typer.Option(
            help="MiPro: Minibatch size for faster evaluation (lower = faster but less stable, typical: 20-50)",
        ),
    ] = 35,
    mipro_minibatch_full_eval_steps: Annotated[
        int,
        typer.Option(
            help="MiPro: Frequency of full validation evaluations (higher = faster, typical: 5-10)",
        ),
    ] = 5,
    mipro_program_aware_proposer: Annotated[
        bool,
        typer.Option(
            help="MiPro: Enable program-aware instruction generation (disable for faster tuning)",
        ),
    ] = True,
    mipro_data_aware_proposer: Annotated[
        bool,
        typer.Option(
            help="MiPro: Enable data-aware instruction generation (disable for faster tuning)",
        ),
    ] = True,
    mipro_tip_aware_proposer: Annotated[
        bool,
        typer.Option(
            help="MiPro: Enable tip-based instruction generation (disable for faster tuning)",
        ),
    ] = True,
    mipro_fewshot_aware_proposer: Annotated[
        bool,
        typer.Option(
            help="MiPro: Enable few-shot aware instruction generation (disable for faster tuning)",
        ),
    ] = True,
) -> None:
    """Tune DSPy classification model with automatic validation evaluation.

    Examples:
        # Basic tuning with defaults
        symptom-diagnosis-explorer classify tune

        # Custom optimizer and dataset size
        symptom-diagnosis-explorer classify tune --optimizer mipro --train-size 100 --val-size 20

        # Different LLM model
        symptom-diagnosis-explorer classify tune --lm-model ollama/mistral:7b
    """
    console.print("[bold]Tuning classification model...[/bold]")

    try:
        # Handle mipro_auto conversion: "none" string -> None
        mipro_auto_value = None if mipro_auto == "none" else mipro_auto

        # Construct full experiment name: /symptom-diagnosis-explorer/{project}/{experiment-name}
        full_experiment_name = (
            f"/symptom-diagnosis-explorer/{project}/{experiment_name}"
        )

        # Create request and execute command
        request = TuneRequest(
            optimizer=optimizer,
            train_size=train_size,
            val_size=val_size,
            model_name=model_name,
            experiment_name=full_experiment_name,
            experiment_project=project,
            lm_model=lm_model,
            num_threads=num_threads,
            mlflow_tracking_uri=mlflow_tracking_uri,
            bootstrap_max_bootstrapped_demos=bootstrap_max_bootstrapped_demos,
            bootstrap_max_labeled_demos=bootstrap_max_labeled_demos,
            mipro_auto=mipro_auto_value,
            mipro_minibatch_size=mipro_minibatch_size,
            mipro_minibatch_full_eval_steps=mipro_minibatch_full_eval_steps,
            mipro_program_aware_proposer=mipro_program_aware_proposer,
            mipro_data_aware_proposer=mipro_data_aware_proposer,
            mipro_tip_aware_proposer=mipro_tip_aware_proposer,
            mipro_fewshot_aware_proposer=mipro_fewshot_aware_proposer,
        )
        command = TuneCommand(request)
        response = command.execute()

        # Display results
        console.print("\n[bold green]✓ Tuning complete![/bold green]\n")

        # Metrics table
        metrics_table = Table(title="Training Metrics")
        metrics_table.add_column("Metric", style="bold magenta")
        metrics_table.add_column("Value", style="cyan")

        metrics_table.add_row(
            "Train Accuracy", f"{response.metrics.train_accuracy:.4f}"
        )
        metrics_table.add_row(
            "Validation Accuracy", f"{response.metrics.validation_accuracy:.4f}"
        )
        metrics_table.add_row(
            "Train Examples", str(response.metrics.num_train_examples)
        )
        metrics_table.add_row(
            "Validation Examples", str(response.metrics.num_val_examples)
        )

        console.print(metrics_table)

        # Model info
        console.print("\n[bold]Model Registry:[/bold]")
        console.print(f"  Name: {response.model_info.name}")
        console.print(f"  Version: {response.model_info.version}")
        console.print(f"  Run ID: {response.run_id}")

    except Exception as e:
        console.print(f"[red]Error tuning model: {e}[/red]")
        raise typer.Exit(1)


@classify_app.command("evaluate")
def classify_evaluate(
    model_name: Annotated[
        str,
        typer.Option(
            help="Model name in MLFlow registry",
        ),
    ] = "symptom-classifier",
    model_version: Annotated[
        Optional[str],
        typer.Option(
            help="Model version (None = latest)",
        ),
    ] = None,
    split: Annotated[
        Literal["train", "validation", "test"],
        typer.Option(
            help="Dataset split (train/validation/test)",
        ),
    ] = "test",
    eval_size: Annotated[
        Optional[int],
        typer.Option(
            help="Limit evaluation examples (None = use all)",
        ),
    ] = None,
    project: Annotated[
        str,
        typer.Option(
            help="Project identifier for MLFlow (e.g., 1-dspy)",
        ),
    ] = "default",
    experiment_name: Annotated[
        str,
        typer.Option(
            help="Experiment name for MLFlow (auto-constructs /symptom-diagnosis-explorer/{project}/{experiment-name})",
        ),
    ] = "evaluate",
    mlflow_tracking_uri: Annotated[
        str,
        typer.Option(
            help="MLFlow tracking server URI",
        ),
    ] = "http://localhost:5001",
) -> None:
    """Evaluate saved model on specified dataset split.

    Examples:
        # Evaluate latest version on test set
        symptom-diagnosis-explorer classify evaluate

        # Evaluate specific version on validation set
        symptom-diagnosis-explorer classify evaluate --model-version 2 --split validation

        # Evaluate on first 10 test examples
        symptom-diagnosis-explorer classify evaluate --eval-size 10
    """
    console.print(f"[bold]Evaluating model on {split} set...[/bold]")

    try:
        # Construct full experiment name
        full_experiment_name = (
            f"/symptom-diagnosis-explorer/{project}/{experiment_name}"
        )

        # Create request and execute command
        request = EvaluateRequest(
            model_name=model_name,
            model_version=model_version,
            split=split,
            eval_size=eval_size,
            experiment_name=full_experiment_name,
            experiment_project=project,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        command = EvaluateCommand()
        response = command.execute(request)

        # Display results
        console.print("\n[bold green]✓ Evaluation complete![/bold green]\n")

        # Results table
        results_table = Table(title=f"Evaluation Results - {split.capitalize()} Set")
        results_table.add_column("Metric", style="bold magenta")
        results_table.add_column("Value", style="cyan")

        results_table.add_row("Model", model_name)
        results_table.add_row("Version", model_version if model_version else "latest")
        results_table.add_row("Split", response.split)
        results_table.add_row("Accuracy", f"{response.accuracy:.4f}")
        results_table.add_row("Examples", str(response.num_examples))
        results_table.add_row("Run ID", response.run_id)

        console.print(results_table)

    except Exception as e:
        console.print(f"[red]Error evaluating model: {e}[/red]")
        raise typer.Exit(1)


@classify_app.command("list-models")
def classify_list_models(
    name_filter: Annotated[
        Optional[str],
        typer.Option(
            help="Filter models by name (substring match)",
        ),
    ] = None,
) -> None:
    """List models in MLFlow registry.

    Examples:
        # List all models
        symptom-diagnosis-explorer classify list-models

        # Filter by name
        symptom-diagnosis-explorer classify list-models --name-filter symptom
    """
    console.print("[bold]Querying MLFlow model registry...[/bold]")

    try:
        # Create request and execute command
        request = ListModelsRequest(name_filter=name_filter)
        command = ListModelsCommand()
        response = command.execute(request)

        # Display results
        if response.total_count == 0:
            console.print("\n[yellow]No models found in registry.[/yellow]")
            return

        console.print(
            f"\n[bold green]Found {response.total_count} model(s)[/bold green]\n"
        )

        # Models table
        models_table = Table(title="Registered Models")
        models_table.add_column("Name", style="bold cyan")
        models_table.add_column("Version", style="magenta")
        models_table.add_column("Aliases", style="yellow")
        models_table.add_column("Created", style="green")
        models_table.add_column("Val Accuracy", style="blue")

        for _, row in response.models.iterrows():
            # Get validation accuracy from metrics if available
            val_acc = row["metrics"].get("validation_accuracy", None)
            val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"

            # Format aliases
            aliases_str = ", ".join(row["aliases"]) if row["aliases"] else "None"

            models_table.add_row(
                row["name"],
                str(row["version"]),
                aliases_str,
                row["creation_time"].strftime("%Y-%m-%d %H:%M"),
                val_acc_str,
            )

        console.print(models_table)

    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        raise typer.Exit(1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
