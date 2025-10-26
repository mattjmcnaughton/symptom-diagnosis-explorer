"""Shared fixtures and utilities for integration tests.

This module provides reusable test fixtures for:
- Ollama availability checks
- MLFlow temporary directories
- Common test model selection
"""

import shutil
from pathlib import Path

import ollama
import pytest


def check_ollama_available() -> bool:
    """Check if Ollama is installed and running.

    Returns:
        True if Ollama is available, False otherwise.
    """
    try:
        ollama.list()
        return True
    except Exception:
        return False


def check_ollama_model_available(model_name: str) -> bool:
    """Check if an Ollama model is available locally.

    Args:
        model_name: Name of the Ollama model (e.g., "qwen3:8b").

    Returns:
        True if model is available, False otherwise.
    """
    try:
        models = ollama.list()
        return any(model_name in model.model for model in models.models)
    except Exception:
        return False


def check_all_required_models(required_models: list[str]) -> tuple[bool, list[str]]:
    """Check if all required Ollama models are available.

    Args:
        required_models: List of required model names.

    Returns:
        Tuple of (all_available, missing_models).
    """
    if not check_ollama_available():
        return False, ["Ollama not installed or not running"]

    missing = [
        model for model in required_models if not check_ollama_model_available(model)
    ]
    return len(missing) == 0, missing


@pytest.fixture(scope="module")
def test_model_name() -> str:
    """Return the Ollama model to use for testing.

    Uses qwen3:1.7b as the default test model.
    """
    return "qwen3:1.7b"


@pytest.fixture(scope="module")
def mlflow_test_dir(tmp_path_factory) -> Path:
    """Create a temporary directory for MLFlow tracking.

    Args:
        tmp_path_factory: Pytest fixture for creating temp directories.

    Returns:
        Path to temporary MLFlow directory.
    """
    mlflow_dir = tmp_path_factory.mktemp("mlflow_test")
    yield mlflow_dir
    # Cleanup after all tests in module
    if mlflow_dir.exists():
        shutil.rmtree(mlflow_dir)
