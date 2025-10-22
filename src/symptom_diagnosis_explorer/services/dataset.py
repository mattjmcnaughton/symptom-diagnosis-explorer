"""Service for loading and managing the symptom diagnosis dataset from Hugging Face."""

from typing import Optional

import pandas as pd
import pandera.typing as pat
from datasets import load_dataset

from symptom_diagnosis_explorer.models import DatasetSplit, SymptomDiagnosisDatasetDF


class DatasetService:
    """Service for accessing the gretelai/symptom_to_diagnosis dataset from Hugging Face.

    This class provides methods to load and access the dataset, converting it to
    pandas DataFrames for easy data manipulation and analysis.
    """

    DATASET_NAME = "gretelai/symptom_to_diagnosis"

    def __init__(self) -> None:
        """Initialize the dataset service."""
        self._dataset = None
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Load the dataset from Hugging Face Hub."""
        self._dataset = load_dataset(self.DATASET_NAME)

    def _prepare_and_validate_dataframe(
        self, df: pd.DataFrame
    ) -> pat.DataFrame[SymptomDiagnosisDatasetDF]:
        """Prepare and validate a raw dataset DataFrame.

        This helper method renames columns from the original HuggingFace format
        (input_text, output_text) to our schema format (symptoms, diagnosis) and
        validates the result against the Pandera schema.

        Args:
            df: Raw pandas DataFrame from the dataset with input_text/output_text columns.

        Returns:
            Validated DataFrame with renamed columns conforming to SymptomDiagnosisDatasetDF.
        """
        # Rename columns to match our schema
        df = df.rename(columns={"input_text": "symptoms", "output_text": "diagnosis"})
        # Validate against schema
        return SymptomDiagnosisDatasetDF.validate(df)

    def get_train_dataframe(self) -> pat.DataFrame[SymptomDiagnosisDatasetDF]:
        """Get the training split as a pandas DataFrame.

        Returns:
            Validated pandas DataFrame containing the training data with columns:
            symptoms, diagnosis.

        Raises:
            RuntimeError: If the dataset has not been loaded yet.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        if self._train_df is None:
            df = self._dataset["train"].to_pandas()
            self._train_df = self._prepare_and_validate_dataframe(df)

        return self._train_df

    def get_test_dataframe(self) -> pat.DataFrame[SymptomDiagnosisDatasetDF]:
        """Get the test split as a pandas DataFrame.

        Returns:
            Validated pandas DataFrame containing the test data with columns:
            symptoms, diagnosis.

        Raises:
            RuntimeError: If the dataset has not been loaded yet.
            ValueError: If the dataset does not contain a 'test' split.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        if self._test_df is None:
            # Check if test split exists
            if "test" not in self._dataset:
                raise ValueError("Dataset does not contain a 'test' split.")

            # Convert the test split to pandas
            df = self._dataset["test"].to_pandas()
            self._test_df = self._prepare_and_validate_dataframe(df)

        return self._test_df

    def get_all_data_dataframe(self) -> pat.DataFrame[SymptomDiagnosisDatasetDF]:
        """Get all available splits concatenated into a single pandas DataFrame.

        Returns:
            Validated pandas DataFrame containing all data from all splits with columns:
            symptoms, diagnosis.

        Raises:
            RuntimeError: If the dataset has not been loaded yet.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        dataframes = []

        # Iterate through all splits in the dataset
        for split_name in self._dataset.keys():
            df = self._dataset[split_name].to_pandas()
            dataframes.append(self._prepare_and_validate_dataframe(df))

        # Concatenate all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Validate against schema (again, to ensure concat didn't break anything)
        return SymptomDiagnosisDatasetDF.validate(combined_df)

    def get_dataset_info(self) -> dict:
        """Get information about the dataset.

        Returns:
            Dictionary containing dataset information such as splits, features, etc.

        Raises:
            RuntimeError: If the dataset has not been loaded yet.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        info = {
            "dataset_name": self.DATASET_NAME,
            "splits": list(self._dataset.keys()),
        }

        # Add information about each split
        for split_name in self._dataset.keys():
            split_data = self._dataset[split_name]
            info[f"{split_name}_num_rows"] = len(split_data)
            info[f"{split_name}_features"] = list(split_data.features.keys())

        return info


def load_symptom_diagnosis_data(
    split: Optional[DatasetSplit | str] = None,
) -> pat.DataFrame[SymptomDiagnosisDatasetDF]:
    """Convenience function to quickly load the symptom diagnosis dataset.

    Args:
        split: The split to load (DatasetSplit.TRAIN, DatasetSplit.TEST, DatasetSplit.ALL,
               or their string equivalents "train", "test", "all").
               If None, loads all splits.

    Returns:
        Validated pandas DataFrame containing the requested data with columns:
        symptoms, diagnosis.

    Raises:
        ValueError: If the split is not found in the dataset.

    Example:
        >>> df = load_symptom_diagnosis_data(split=DatasetSplit.TRAIN)
        >>> print(df.head())
        >>> df = load_symptom_diagnosis_data(split="train")
        >>> print(df.head())
    """
    service = DatasetService()
    service.load()

    # Convert string to enum if needed
    if isinstance(split, str):
        try:
            split = DatasetSplit(split)
        except ValueError:
            available_splits = list(service._dataset.keys())
            raise ValueError(
                f"Split '{split}' not found in dataset. Available splits: {available_splits}"
            )

    if split == DatasetSplit.TRAIN:
        return service.get_train_dataframe()
    elif split == DatasetSplit.TEST:
        return service.get_test_dataframe()
    elif split == DatasetSplit.ALL or split is None:
        return service.get_all_data_dataframe()

    # This should never be reached, but just in case
    raise ValueError(f"Invalid split: {split}")
