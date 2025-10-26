# Install dependencies
install:
     uv sync --group dev

dev:
     uv run txtpack

lint:
     uv run ruff check ./src

lint-fix:
     uv run ruff check ./src --fix

format:
     uv run ruff format ./src

format-check:
     uv run ruff format --check ./src

typecheck:
     uv run ty check src

test *args:
     uv run pytest {{args}}

test-unit:
     uv run pytest tests/unit/

test-integration:
     uv run pytest tests/integration/

test-e2e:
     uv run pytest tests/e2e/

ci: lint format-check typecheck test

# Create Jupyter kernel for this environment
kernel:
    uv run python -m ipykernel install --user --name=symptom-diagnosis-explorer --display-name="Symptom Diagnosis Explorer"

# Launch Jupyter Lab
notebook:
    uv run jupyter lab

# Launch MLflow UI
mlflow:
    uvx mlflow ui --backend-store-uri .mlflow --default-artifact-root .mlflow/artifacts --host 0.0.0.0 --port 5001

# Strip output from all Jupyter notebooks
nbstripout:
    uv run nbstripout **/*.ipynb
