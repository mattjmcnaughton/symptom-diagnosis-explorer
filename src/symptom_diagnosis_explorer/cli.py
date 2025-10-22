"""Command-line interface for the Symptom Diagnosis Explorer."""

import typer
from typing_extensions import Annotated

app = typer.Typer(
    name="symptom-diagnosis-explorer",
    help="Explore and analyze symptom diagnosis using ML models.",
    add_completion=False,
)


@app.command()
def hello(
    name: Annotated[str, typer.Option(help="Name to greet")] = "World",
) -> None:
    """Say hello to someone."""
    typer.echo(f"Hello {name}!")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
