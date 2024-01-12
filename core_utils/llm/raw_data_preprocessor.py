"""
Module with description of abstract raw data preprocessor.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import pandas as pd


class ColumnNames(Enum):
    """
    Column names for preprocessed DataFrame.
    """

    SOURCE_NLI = ['premise', 'hypothesis']
    SOURCE = 'source'
    TARGET = 'target'
    PREDICTION = 'predictions'
    QUESTION = 'question'
    CONTEXT = 'context'
    PREMISE = 'premise'
    HYPOTHESIS = 'hypothesis'

    def __str__(self) -> Any:
        return self.value


class AbstractRawDataPreprocessor(ABC):
    """
    Abstract Raw Data Preprocessor.
    """
    _data: pd.DataFrame | None

    def __init__(self, raw_data: pd.DataFrame):
        self._raw_data = raw_data
        self._data = None

    @abstractmethod
    def analyze(self) -> dict:
        """
        Abstract method for dataset analysis.
        """

    @abstractmethod
    def transform(self) -> None:
        """
        Abstract method for dataset preprocessing.
        """

    @property
    def data(self) -> pd.DataFrame | None:
        """
        Property for original dataset.
        """
        return self._data
