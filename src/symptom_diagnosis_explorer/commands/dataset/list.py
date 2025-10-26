"""Dataset list command."""

import pandas as pd
from pydantic import BaseModel, Field

from symptom_diagnosis_explorer.services.dataset import DatasetService


class DatasetListRequest(BaseModel):
    """Request model for listing dataset rows."""

    split: str = Field(
        default="all",
        description="Dataset split to display (train/validation/test/all)",
    )
    rows: int = Field(
        default=5,
        description="Number of rows to display",
        gt=0,
    )


class DatasetListResponse(BaseModel):
    """Response model for listing dataset rows."""

    df: pd.DataFrame = Field(
        description="DataFrame containing the requested rows",
    )
    total_rows: int = Field(
        description="Total number of rows in the full dataset split",
    )

    model_config = {"arbitrary_types_allowed": True}


class DatasetListCommand:
    """Command to list rows from the dataset."""

    def __init__(self) -> None:
        """Initialize the command."""
        self.service = DatasetService()

    def execute(self, request: DatasetListRequest) -> DatasetListResponse:
        """Execute the dataset list command.

        Args:
            request: The request containing split and row count parameters.

        Returns:
            Response containing the dataframe and total row count.

        Raises:
            RuntimeError: If the dataset cannot be loaded.
            ValueError: If the split is invalid.
        """
        self.service.load()

        # Get the appropriate dataframe based on split
        if request.split == "train":
            df = self.service.get_train_dataframe()
        elif request.split == "validation":
            df = self.service.get_validation_dataframe()
        elif request.split == "test":
            df = self.service.get_test_dataframe()
        elif request.split == "all":
            df = self.service.get_all_data_dataframe()
        else:
            raise ValueError(
                f"Invalid split '{request.split}'. Choose from: train, validation, test, all"
            )

        # Get the total row count
        total_rows = len(df)

        # Limit to requested number of rows
        limited_df = df.head(request.rows)

        return DatasetListResponse(df=limited_df, total_rows=total_rows)
