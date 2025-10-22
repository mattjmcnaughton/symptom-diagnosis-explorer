"""Command-line interface for the Symptom Diagnosis Explorer."""

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from symptom_diagnosis_explorer.commands import (
    DatasetListCommand,
    DatasetListRequest,
    DatasetSummaryCommand,
    DatasetSummaryRequest,
)

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


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
