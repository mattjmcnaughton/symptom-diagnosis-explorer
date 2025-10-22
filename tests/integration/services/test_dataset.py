"""Integration tests for the DatasetService."""

import pandas as pd
import pytest

from symptom_diagnosis_explorer.models import (
    DatasetSplit,
    DiagnosisType,
    SymptomDiagnosisDatasetDF,
    SymptomDiagnosisExample,
)
from symptom_diagnosis_explorer.services.dataset import (
    DatasetService,
    load_symptom_diagnosis_data,
)


class TestDatasetService:
    """Integration tests for DatasetService class."""

    @pytest.fixture
    def service(self):
        """Create and load a DatasetService instance."""
        svc = DatasetService()
        svc.load()
        return svc

    def test_load_dataset(self):
        """Test that dataset can be loaded successfully."""
        service = DatasetService()
        assert service._dataset is None

        service.load()
        assert service._dataset is not None
        assert "train" in service._dataset

    def test_get_train_dataframe(self, service):
        """Test getting training data with proper schema validation."""
        train_df = service.get_train_dataframe()

        # Verify it's a DataFrame
        assert isinstance(train_df, pd.DataFrame)

        # Verify columns match our schema
        assert "symptoms" in train_df.columns
        assert "diagnosis" in train_df.columns

        # Verify non-empty
        assert len(train_df) > 0

        # Verify all diagnoses are valid
        for diagnosis in train_df["diagnosis"].unique():
            assert diagnosis in [d.value for d in DiagnosisType]

        # Verify no null values in required columns
        assert train_df["symptoms"].notna().all()
        assert train_df["diagnosis"].notna().all()

    def test_get_test_dataframe(self, service):
        """Test getting test data with proper schema validation."""
        test_df = service.get_test_dataframe()

        # Verify it's a DataFrame
        assert isinstance(test_df, pd.DataFrame)

        # Verify columns match our schema
        assert "symptoms" in test_df.columns
        assert "diagnosis" in test_df.columns

        # Verify non-empty
        assert len(test_df) > 0

        # Verify all diagnoses are valid
        for diagnosis in test_df["diagnosis"].unique():
            assert diagnosis in [d.value for d in DiagnosisType]

        # Verify no null values in required columns
        assert test_df["symptoms"].notna().all()
        assert test_df["diagnosis"].notna().all()

    def test_get_all_data_dataframe(self, service):
        """Test getting all data from both splits concatenated."""
        all_df = service.get_all_data_dataframe()

        # Verify it's a DataFrame
        assert isinstance(all_df, pd.DataFrame)

        # Verify columns match our schema
        assert "symptoms" in all_df.columns
        assert "diagnosis" in all_df.columns

        # Verify non-empty
        assert len(all_df) > 0

        # Verify all diagnoses are valid
        for diagnosis in all_df["diagnosis"].unique():
            assert diagnosis in [d.value for d in DiagnosisType]

        # Verify no null values in required columns
        assert all_df["symptoms"].notna().all()
        assert all_df["diagnosis"].notna().all()

        # Verify train + test = all
        train_df = service.get_train_dataframe()
        test_df = service.get_test_dataframe()
        assert len(all_df) == len(train_df) + len(test_df)

    def test_get_dataset_info(self, service):
        """Test getting dataset metadata."""
        info = service.get_dataset_info()

        assert "dataset_name" in info
        assert info["dataset_name"] == "gretelai/symptom_to_diagnosis"
        assert "splits" in info
        assert "train" in info["splits"]
        assert "train_num_rows" in info
        assert info["train_num_rows"] > 0

    def test_caching_behavior(self, service):
        """Test that DataFrames are cached after first load."""
        # First call
        train_df_1 = service.get_train_dataframe()

        # Second call should return the same cached object
        train_df_2 = service.get_train_dataframe()

        # Verify they're the same object (cached)
        assert train_df_1 is train_df_2

    def test_pandera_validation(self, service):
        """Test that Pandera schema validation is applied."""
        train_df = service.get_train_dataframe()

        # This should not raise an exception if validation passes
        validated_df = SymptomDiagnosisDatasetDF.validate(train_df)

        # Verify it returns a DataFrame
        assert isinstance(validated_df, pd.DataFrame)


class TestSymptomDiagnosisExample:
    """Integration tests for SymptomDiagnosisExample Pydantic model."""

    @pytest.fixture
    def sample_row(self):
        """Create a sample DataFrame row for testing."""
        service = DatasetService()
        service.load()
        train_df = service.get_train_dataframe()
        return train_df.iloc[0]

    def test_create_example_from_dataframe_row(self, sample_row):
        """Test creating a Pydantic model from a DataFrame row."""
        example = SymptomDiagnosisExample(
            symptoms=sample_row["symptoms"], diagnosis=sample_row["diagnosis"]
        )

        assert isinstance(example, SymptomDiagnosisExample)
        assert isinstance(example.symptoms, str)
        assert isinstance(example.diagnosis, DiagnosisType)
        assert len(example.symptoms) > 0

    def test_example_with_alias(self, sample_row):
        """Test that aliases (input_text, output_text) work correctly."""
        # Use model_validate to bypass type checking for aliases
        example = SymptomDiagnosisExample.model_validate(
            {
                "input_text": sample_row["symptoms"],
                "output_text": sample_row["diagnosis"],
            }
        )

        # Should be accessible via field name
        assert example.symptoms == sample_row["symptoms"]
        assert example.diagnosis.value == sample_row["diagnosis"]

    def test_label_property(self, sample_row):
        """Test that the label property returns the diagnosis."""
        example = SymptomDiagnosisExample(
            symptoms=sample_row["symptoms"], diagnosis=sample_row["diagnosis"]
        )

        # label should be an alias for diagnosis
        assert example.label == example.diagnosis
        assert isinstance(example.label, DiagnosisType)

    def test_whitespace_validation(self):
        """Test that whitespace is stripped from symptoms."""
        example = SymptomDiagnosisExample(
            symptoms="  I have a headache  ", diagnosis=DiagnosisType.MIGRAINE
        )

        # Whitespace should be stripped
        assert example.symptoms == "I have a headache"

    def test_invalid_diagnosis_raises_error(self):
        """Test that invalid diagnosis values raise validation errors."""
        with pytest.raises(ValueError):
            SymptomDiagnosisExample.model_validate(
                {"symptoms": "I feel sick", "diagnosis": "invalid_diagnosis"}
            )


class TestLoadSymptomDiagnosisData:
    """Integration tests for the convenience function."""

    def test_load_train_split(self):
        """Test loading train split via convenience function."""
        df = load_symptom_diagnosis_data(split=DatasetSplit.TRAIN)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "symptoms" in df.columns
        assert "diagnosis" in df.columns

    def test_load_test_split(self):
        """Test loading test split via convenience function."""
        df = load_symptom_diagnosis_data(split=DatasetSplit.TEST)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "symptoms" in df.columns
        assert "diagnosis" in df.columns

    def test_load_all_splits(self):
        """Test loading all splits via convenience function."""
        df = load_symptom_diagnosis_data(split=DatasetSplit.ALL)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "symptoms" in df.columns
        assert "diagnosis" in df.columns

    def test_load_all_splits_with_none(self):
        """Test loading all splits when split is None."""
        df = load_symptom_diagnosis_data(split=None)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "symptoms" in df.columns
        assert "diagnosis" in df.columns


class TestDiagnosisTypeEnum:
    """Integration tests for DiagnosisType enum."""

    def test_all_enum_values_exist_in_dataset(self):
        """Test that all enum values correspond to actual diagnoses in the dataset."""
        service = DatasetService()
        service.load()
        all_df = service.get_all_data_dataframe()

        # Get all unique diagnoses from dataset
        dataset_diagnoses = set(all_df["diagnosis"].unique())

        # Get all enum values
        enum_values = {d.value for d in DiagnosisType}

        # Every enum value should exist in the dataset
        assert enum_values == dataset_diagnoses

    def test_enum_count(self):
        """Test that we have exactly 22 diagnoses."""
        assert len(DiagnosisType) == 22
