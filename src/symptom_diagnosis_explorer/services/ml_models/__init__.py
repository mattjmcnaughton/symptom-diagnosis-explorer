"""ML Models module for multi-framework classification support.

This module provides a unified interface for different ML frameworks
(DSPy, LangChain, Pydantic-AI, etc.) through abstract base classes and a registry pattern.
"""

from symptom_diagnosis_explorer.services.ml_models.base import BaseModelService
from symptom_diagnosis_explorer.services.ml_models.dspy import DSPyModelService
from symptom_diagnosis_explorer.services.ml_models.langchain import (
    LangChainModelService,
)
from symptom_diagnosis_explorer.services.ml_models.pydantic_ai import (
    PydanticAIModelService,
)
from symptom_diagnosis_explorer.services.ml_models.registry import FrameworkRegistry

__all__ = [
    "BaseModelService",
    "DSPyModelService",
    "LangChainModelService",
    "PydanticAIModelService",
    "FrameworkRegistry",
]
