"""Dataset management models."""

from enum import Enum


class DatasetSplit(str, Enum):
    """Enum for dataset splits."""

    TRAIN = "train"
    TEST = "test"
    ALL = "all"
