"""
Neural machine translation module.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Iterator, Sequence

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
from datasets import load_dataset
import pandas as pd


class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        self._raw_data = load_dataset("HuggingFaceH4/no_robots", split='train_sft').to_pandas()


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """
    @report_time
    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        properties_dict = {
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": self._raw_data.drop(["messages"], axis=1).duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isna().sum().sum()
        }
        self._raw_data = self._raw_data.dropna()
        properties_dict["dataset_sample_min_len"] = min(len(min(self._raw_data["prompt"], key=len)),
                                                        self.get_min_len())
        properties_dict["dataset_sample_max_len"] = max(len(max(self._raw_data["prompt"], key=len)),
                                                        self.get_max_len())
        return properties_dict

    def get_min_len(self):
        min_len = (len(self._raw_data["messages"][0][0]["content"])
                   + len(self._raw_data["messages"][0][1]["content"]))
        index = -1
        for row in self._raw_data["messages"]:
            index += 1
            if len(row[0]["content"]) + len(row[1]["content"]) < min_len:
                min_len = len(row[0]["content"]) + len(row[1]["content"])
        return min_len

    def get_max_len(self):
        max_len = (len(self._raw_data["messages"][0][0]["content"])
                   + len(self._raw_data["messages"][0][1]["content"]))
        index = -1
        for row in self._raw_data["messages"]:
            index += 1
            if len(row[0]["content"]) + len(row[1]["content"]) > max_len:
                max_len = len(row[0]["content"]) + len(row[1]["content"])
        return max_len


    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """

        self._data = self._raw_data[(self._raw_data.category == 'Closed QA')]
        self._data = self._data.rename(columns={'prompt': 'questions'}, inplace=False)
        self._data[['context', 'answer']] = pd.DataFrame(self._data['messages'].to_list(), index=self._data.index)
        self._data[['answer', 'answer3']] = pd.DataFrame(self._data['answer'].apply(pd.Series),
                                                         index=self._data.index)
        self._data[['context', 'context3']] = pd.DataFrame(self._data['context'].apply(pd.Series),
                                                           index=self._data.index)
        self._data = self._data.drop(['prompt_id', 'category', 'messages', 'answer3', 'context3'], axis=1)
        self._data = self._data.dropna()
        self._data = self._data.drop_duplicates()
        self._data = self._data.reset_index(drop=True)


class TaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
        """

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """


class LLMPipeline(AbstractLLMPipeline):
    """
    A class that initializes a model, analyzes its properties and infers it.
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
        Initialize an instance of LLMPipeline.

        Args:
            model_name (str): The name of the pre-trained model
            dataset (TaskDataset): The dataset used
            max_length (int): The maximum length of generated sequence
            batch_size (int): The size of the batch inside DataLoader
            device (str): The device for inference
        """

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """

    @report_time
    def infer_dataset(self) -> DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """


class TaskEvaluator(AbstractTaskEvaluator):
    """
    A class that compares prediction quality using the specified metric.
    """

    def __init__(self, data_path: Path, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of Evaluator.

        Args:
            data_path (pathlib.Path): Path to predictions
            metrics (Iterable[Metrics]): List of metrics to check
        """

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """


#importer = RawDataImporter('')
#importer.obtain()

# Load model directly
'''from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
question = "Where do I live?"
context = "My name is Tim and I live in Sweden."
tokens = tokenizer(question, context, return_tensors='pt')
print(tokens)'''
