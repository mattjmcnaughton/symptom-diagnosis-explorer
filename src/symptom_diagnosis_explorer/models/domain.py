"""Core domain models for symptom diagnosis.

This module contains all the fundamental domain models that define the symptom
diagnosis problem: diagnosis types, examples, dataset schemas, and DSPy signatures.
"""

from enum import Enum

import dspy
import pandera.pandas as pa
from pandera.engines.pandas_engine import PydanticModel
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DiagnosisType(str, Enum):
    """Enumeration of all possible medical diagnoses in the dataset."""

    ALLERGY = "allergy"
    ARTHRITIS = "arthritis"
    BRONCHIAL_ASTHMA = "bronchial asthma"
    CERVICAL_SPONDYLOSIS = "cervical spondylosis"
    CHICKEN_POX = "chicken pox"
    COMMON_COLD = "common cold"
    DENGUE = "dengue"
    DIABETES = "diabetes"
    DRUG_REACTION = "drug reaction"
    FUNGAL_INFECTION = "fungal infection"
    GASTROESOPHAGEAL_REFLUX_DISEASE = "gastroesophageal reflux disease"
    HYPERTENSION = "hypertension"
    IMPETIGO = "impetigo"
    JAUNDICE = "jaundice"
    MALARIA = "malaria"
    MIGRAINE = "migraine"
    PEPTIC_ULCER_DISEASE = "peptic ulcer disease"
    PNEUMONIA = "pneumonia"
    PSORIASIS = "psoriasis"
    TYPHOID = "typhoid"
    URINARY_TRACT_INFECTION = "urinary tract infection"
    VARICOSE_VEINS = "varicose veins"


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


class SymptomDiagnosisSignature(dspy.Signature):
    """DSPy signature for symptom-to-diagnosis classification.

    This signature defines the input/output interface for the language model task.
    DSPy uses this to automatically generate prompts and parse responses.

    Attributes:
        symptoms: Patient's description of their symptoms (input).
        diagnosis: The predicted medical diagnosis (output, constrained to valid types).
    """

    symptoms: str = dspy.InputField(desc="Patient symptoms description")
    diagnosis: DiagnosisType = dspy.OutputField(desc="Diagnosis category")
