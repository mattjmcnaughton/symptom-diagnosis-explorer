"""Dataset domain models."""

from enum import Enum

import pandera.pandas as pa
from pandera.engines.pandas_engine import PydanticModel
from pydantic import BaseModel, ConfigDict, Field, field_validator

from symptom_diagnosis_explorer.models.diagnosis import DiagnosisType


class DatasetSplit(str, Enum):
    """Enum for dataset splits."""

    TRAIN = "train"
    TEST = "test"
    ALL = "all"


class SymptomDiagnosisExample(BaseModel):
    """Pydantic model representing a single symptom-diagnosis example.

    This model provides type-safe access to individual records from the dataset,
    with both domain-specific field names (symptoms, diagnosis) and compatibility
    with the original dataset field names (input_text, output_text) via aliases.

    Attributes:
        symptoms: Patient's description of their symptoms (input text).
        diagnosis: The medical diagnosis for the symptoms (output label).

    Examples:
        >>> example = SymptomDiagnosisExample(
        ...     input_text="I have a fever and headache",
        ...     output_text="common cold"
        ... )
        >>> print(example.symptoms)
        "I have a fever and headache"
        >>> print(example.label)  # ML-friendly alias
        "common cold"
    """

    symptoms: str = Field(
        alias="input_text",
        description="Patient's description of symptoms",
        min_length=1,
    )
    diagnosis: DiagnosisType = Field(
        alias="output_text",
        description="Medical diagnosis for the symptoms",
    )

    model_config = ConfigDict(
        populate_by_name=True,  # Allow using both field name and alias
    )

    @field_validator("symptoms")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading and trailing whitespace from symptoms."""
        return v.strip()

    @property
    def label(self) -> DiagnosisType:
        """ML-friendly alias for diagnosis field.

        Returns:
            The diagnosis value, useful in ML contexts where "label" is conventional.
        """
        return self.diagnosis


class SymptomDiagnosisDatasetDF(pa.DataFrameModel):
    """Pandera schema for a DataFrame of symptom-diagnosis examples.

    This schema validates that DataFrames conform to the expected structure
    by validating each row against the SymptomDiagnosisExample Pydantic model.

    Each row is validated as a SymptomDiagnosisExample with:
        symptoms: Patient symptom descriptions (non-empty strings).
        diagnosis: Medical diagnoses (must be valid DiagnosisType values).

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "symptoms": ["I have a fever"],
        ...     "diagnosis": ["common cold"],
        ... })
        >>> validated_df = SymptomDiagnosisDatasetDF.validate(df)
    """

    class Config:
        """Pandera configuration."""

        dtype = PydanticModel(SymptomDiagnosisExample)
        coerce = True  # Required for PydanticModel
