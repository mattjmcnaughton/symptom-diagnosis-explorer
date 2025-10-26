"""Classify list-models command."""

import pandas as pd
from pydantic import BaseModel, Field

from symptom_diagnosis_explorer.models.model_development import ClassificationConfig
from symptom_diagnosis_explorer.services.model_development import (
    ModelDevelopmentService,
)


class ListModelsRequest(BaseModel):
    """Request model for listing registered models."""

    name_filter: str | None = Field(
        default=None,
        description="Filter models by name (substring match)",
    )


class ListModelsResponse(BaseModel):
    """Response model for list-models command."""

    models: pd.DataFrame = Field(
        description="DataFrame containing registered models",
    )
    total_count: int = Field(
        ge=0,
        description="Total number of models found",
    )

    model_config = {"arbitrary_types_allowed": True}


class ListModelsCommand:
    """Command to list registered models from MLFlow registry."""

    def __init__(self) -> None:
        """Initialize the command."""
        # Use default configuration for listing models
        config = ClassificationConfig()
        self.service = ModelDevelopmentService(config)

    def execute(self, request: ListModelsRequest) -> ListModelsResponse:
        """Execute the list-models command.

        Args:
            request: List models request with optional name filter.

        Returns:
            Response containing DataFrame of models and total count.
        """
        # Call service to list models
        models_df = self.service.list_models(name_filter=request.name_filter)

        return ListModelsResponse(
            models=models_df,
            total_count=len(models_df),
        )
