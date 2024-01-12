"""
Module with description of abstract data importer.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class AbstractRawDataImporter(ABC):
    """
    Abstract Raw Data Importer.
    """

    _local_path: Path | None
    _raw_data: pd.DataFrame | None

    def __init__(self, hf_name: str | None):
        self._hf_name = hf_name
        self._raw_data = None

    @abstractmethod
    def obtain(self) -> None:
        """
        Download a dataset.
        """

    @property
    def raw_data(self) -> pd.DataFrame | None:
        """
        Property for original dataset in a table format.
        """
        return self._raw_data
