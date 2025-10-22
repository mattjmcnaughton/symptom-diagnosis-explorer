"""Diagnosis domain models."""

from enum import Enum


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
