"""
Module with description of abstract llm pipeline.
"""
# pylint: disable=too-few-public-methods, too-many-arguments
from abc import ABC, abstractmethod
from typing import Any, Protocol

import pandas as pd
from torch.utils.data.dataset import Dataset


class HFModelLike(Protocol):
    """
    Protocol definition of HF models.
    """
    def __call__(self, *args: tuple, return_dict: bool = False, **kwargs: dict) -> Any:
        """
        Placeholder to claim HF models are callable.

        Args:
             args (tuple): arbitrary positional arguments
             return_dict (bool): special argument for QA models
             kwargs (dict): arbitrary named arguments
        """


class AbstractLLMPipeline(ABC):
    """
    Abstract LLM Pipeline.
    """

    _model: HFModelLike | None

    def __init__(self, model_name: str, dataset: Dataset, max_length: int,
                 batch_size: int, device: str = 'cpu'):
        self._model_name = model_name
        self._model = None
        self._dataset = dataset
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device

    @abstractmethod
    def infer_sample(self, sample: str) -> str:
        """
        Infer model on a single sample.

        Args:
            sample (str): The given sample for inference with model
        """

    @abstractmethod
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.
        """

    @abstractmethod
    def analyze_model(self) -> dict:
        """
        Analyze model key properties.
        """
