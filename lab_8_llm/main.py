"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called, duplicate-code
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Sequence

from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        self._raw_data = load_dataset(self._hf_name, split='train').to_pandas()


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        properties_dict = {"dataset_number_of_samples": self._raw_data.shape[0],
                           "dataset_columns": self._raw_data.shape[1],
                           "dataset_empty_rows": self._raw_data.isin(['']).sum().sum(),
                           "dataset_duplicates": self._raw_data.duplicated().sum()}

        raw_data = self._raw_data[self._raw_data.context != '']
        properties_dict["dataset_sample_min_len"] = raw_data['instruction'].str.len().min()
        properties_dict["dataset_sample_max_len"] = raw_data['instruction'].str.len().max()

        return properties_dict

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data[(self._raw_data["category"] == 'open_qa')]
        self._data = self._data.rename(columns={'instruction': 'question'}, inplace=False)
        self._data = self._data.rename(columns={'response': 'target'}, inplace=False)
        self._data = self._data.drop(['context', 'category', '__index_level_0__'], axis=1)
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
        self._data = data

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        return self._data.iloc[index]['question'], self._data.iloc[index]['target']

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


class LLMPipeline(AbstractLLMPipeline):
    """
    A class that initializes a model, analyzes its properties and infers it.
    """

    def __init__(self, model_name: str, dataset: TaskDataset, max_length: int, batch_size: int, device: str) -> None:
        """
        Initialize an instance of LLMPipeline.

        Args:
            model_name (str): The name of the pre-trained model
            dataset (TaskDataset): The dataset used
            max_length (int): The maximum length of generated sequence
            batch_size (int): The size of the batch inside DataLoader
            device (str): The device for inference
        """
        super().__init__(model_name, dataset, max_length, batch_size, device)
        self._model_name = model_name
        self._dataset = dataset
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, padding_side='left')
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        model_config = self._model.config
        tensor_data = torch.ones(1, model_config.max_position_embeddings, dtype=torch.long)
        input_data = {"input_ids": tensor_data, "attention_mask": tensor_data}
        torch_summary = summary(self._model, input_data=input_data, verbose=False)

        analyze_dict = {
            "input_shape": {
                'attention_mask': list(torch_summary.input_size),
                'input_ids': list(torch_summary.input_size)
            },
            "embedding_size": model_config.max_position_embeddings,
            "output_shape": torch_summary.summary_list[-1].output_size,
            "num_trainable_params": torch_summary.trainable_params,
            "vocab_size": model_config.vocab_size,
            "size": torch_summary.total_param_bytes,
            "max_context_length": model_config.max_length
        }

        return analyze_dict
    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        if self._model is None:
            return None

        return self._infer_batch((sample,))[0]

    @report_time
    def infer_dataset(self) -> DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        dataloader = DataLoader(self._dataset, batch_size=self._batch_size)
        predictions = []
        for batch in dataloader:
            predictions.extend(self._infer_batch(batch))

        result_dataframe = DataFrame({'target': self._dataset.data["target"],
                                      'predictions': predictions
                                      })

        return result_dataframe

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        self._tokenizer.pad_token = self._tokenizer.eos_token
        tokens = self._tokenizer(sample_batch[0], padding=True, truncation=True, return_tensors='pt')
        model_output = self._model.generate(**tokens, max_length=self._max_length)
        text_output = self._tokenizer.batch_decode(model_output, skip_special_tokens=True)

        return [text_output[len(sample_batch[0][i]) + 1:] for i, prediction in enumerate(text_output)]


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
