"""
Neural machine translation module.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from datasets import load_dataset


try:
    import torch
    from torch.utils.data.dataset import Dataset
except ImportError:
    print('Library "torch" not installed. Failed to import.')
    Dataset = dict
    torch = namedtuple('torch', 'no_grad')(lambda: lambda fn: fn)  # type: ignore

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = dict  # type: ignore

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time


class RawDataImporter(AbstractRawDataImporter):
    """
    Custom implementation of data importer.
    """

    def __init__(self, _hf_name):
        super().__init__(_hf_name)

    @report_time
    def obtain(self) -> None:
        """
        Import dataset.
        """

        raw_dataset = load_dataset("RussianNLP/russian_super_glue", self._hf_name, split='validation')
        self._raw_data = raw_dataset.to_pandas()

    # def raw_data(self) -> DataFrame | None:
    #     return self._raw_data


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    Custom implementation of data preprocessor.
    """

    def __init__(self, _raw_data):
        super().__init__(_raw_data)

    def analyze(self) -> dict:
        """
        Analyze preprocessed dataset.

        Returns:
            dict: dataset key properties.
        """
        analyze_dict = {
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": self._raw_data.duplicated(keep='first').sum(),
            "dataset_empty_rows": self._raw_data.isna().sum().sum(),
            "dataset_sample_min_len": min(self._raw_data['premise'].str.len().min(), self._raw_data['hypothesis'].str.len().min()),
            "dataset_sample_max_len": max(self._raw_data['premise'].str.len().max(), self._raw_data['hypothesis'].str.len().max())
        }
        return analyze_dict

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """


class TaskDataset(Dataset):
    """
    Dataset with translation data.
    """

    def __init__(self, data: DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): original data.
        """

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """

    def __iter__(self) -> Iterator:
        """
        Overriden iter method for static checks.

        Returns:
            Iterator: The iterator instance.
        """

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.
        """


class LLMPipeline(AbstractLLMPipeline):
    """
    Translation model.
    """

    def __init__(
            self,
            model_name: str,
            dataset: TaskDataset,
            max_length: int,
            batch_size: int,
            device: str
    ) -> None:
        """
        Initialize an instance of HelsinkiNLPModel.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (TaskDataset): The dataset to be used for translation.
            max_length (int): The maximum length of generated sequence.
            batch_size (int): The size of the batch inside DataLoader.
            device (str): The device for inference.
        """

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: properties of a model
        """

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.
        """

    @report_time
    def infer_dataset(self) -> DataFrame:
        """
        Translate the dataset sentences.

        Returns:
            list[str]: A list of predictions.
        """

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """


class TaskEvaluator(AbstractTaskEvaluator):
    """
    Evaluator for comparing prediction quality using the specified metric.
    """

    def __init__(self, data_path: Path, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of Evaluator.

        Args:
            data_path (pathlib.Path): Path to predictions.
            metrics (Iterable[Metrics]): List of metrics to check.
        """

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric.
        """
