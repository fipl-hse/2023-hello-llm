"""
Module with description of abstract llm pipeline.
"""
# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code
from abc import ABC, abstractmethod
from typing import Any, Protocol

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = dict  # type: ignore

try:
    from torch.utils.data.dataset import Dataset
except ImportError:
    print('Library "torch" not installed. Failed to import.')
    Dataset = None


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
    def infer_dataset(self) -> DataFrame:
        """
        Infer model on a whole dataset.
        """

    @abstractmethod
    def analyze_model(self) -> dict:
        """
        Analyze model key properties.
        """
