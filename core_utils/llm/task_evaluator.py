"""
Module with description of abstract task evaluator.
"""
# pylint: disable=too-few-public-methods
from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd

from core_utils.llm.metrics import Metrics


class AbstractTaskEvaluator(ABC):
    """
    Abstract Task Evaluator.
    """
    def __init__(self, metrics: Iterable[Metrics]):
        self._metrics = metrics

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """
        Entrypoint for task evaluation versus a number of specified metrics.
        """
