"""Dataset summary command."""

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from symptom_diagnosis_explorer.services.dataset import DatasetService


class DatasetSummaryRequest(BaseModel):
    """Request model for dataset summary statistics."""

    split: str = Field(
        default="all",
        description="Dataset split to summarize (train/test/all)",
    )


class DatasetSummaryResponse(BaseModel):
    """Response model for dataset summary statistics."""

    df: pd.DataFrame = Field(
        description="The full DataFrame for the requested split",
    )
    stats: dict[str, Any] = Field(
        description="Summary statistics for the dataset",
    )

    model_config = {"arbitrary_types_allowed": True}


class DatasetSummaryCommand:
    """Command to get summary statistics for the dataset."""

    def __init__(self) -> None:
        """Initialize the command."""
        self.service = DatasetService()

    def execute(self, request: DatasetSummaryRequest) -> DatasetSummaryResponse:
        """Execute the dataset summary command.

        Args:
            request: The request containing split parameter.

        Returns:
            Response containing the dataframe and summary statistics.

        Raises:
            RuntimeError: If the dataset cannot be loaded.
            ValueError: If the split is invalid.
        """
        self.service.load()

        # Get the appropriate dataframe based on split
        if request.split == "train":
            df = self.service.get_train_dataframe()
        elif request.split == "test":
            df = self.service.get_test_dataframe()
        elif request.split == "all":
            df = self.service.get_all_data_dataframe()
        else:
            raise ValueError(
                f"Invalid split '{request.split}'. Choose from: train, test, all"
            )

        # Compute summary statistics
        stats: dict[str, Any] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "numeric_stats": df.describe(),
            "non_numeric_info": {},
        }

        # Add information about non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
        non_numeric_info: dict[str, dict[str, int]] = {}
        for col in non_numeric_cols:
            non_numeric_info[col] = {
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
            }
        stats["non_numeric_info"] = non_numeric_info

        return DatasetSummaryResponse(df=df, stats=stats)
