"""
Module with description of abstract raw data preprocessor.
"""
# pylint: disable=duplicate-code
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = dict  # type: ignore


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
    _data: DataFrame | None

    def __init__(self, raw_data: DataFrame):
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
    def data(self) -> DataFrame | None:
        """
        Property for original dataset.
        """
        return self._data
