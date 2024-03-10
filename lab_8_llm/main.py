"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called, duplicate-code
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame, read_csv
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchinfo import summary
from transformers import BertForSequenceClassification, BertTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time


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
        dataframe = load_dataset(self._hf_name, split='train').to_pandas()
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError()
        self._raw_data = dataframe


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
        processed_data = self._raw_data.dropna().reset_index(drop=True)
        column_length = processed_data["text"].astype(str).str.len()

        analytics = {
            "dataset_number_of_samples": len(self._raw_data),
            "dataset_columns": len(self._raw_data.columns),
            "dataset_duplicates": self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isnull().any(axis=1).sum(),
            "dataset_sample_min_len": int(column_length.min()),
            "dataset_sample_max_len": int(column_length.max())
        }

        return analytics

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.copy()
        self._data.rename(columns={
                'text': ColumnNames.SOURCE.value,
                'toxic': ColumnNames.TARGET.value
            }, inplace=True)


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
        return (str(self._data[ColumnNames.SOURCE.value].iloc[index]),)

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
        self._model_name = model_name
        self._dataset = dataset
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self._model.to(device)
        self._tokenizer = BertTokenizer.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if self._model is None:
            raise TypeError("self._model is None!")

        noise = torch.ones(1, self._model.config.max_position_embeddings, dtype=torch.long)

        model_info = summary(self._model,
                             input_data={'input_ids': noise, 'attention_mask': noise},
                             device=self._device,
                             verbose=0)

        return {
            "embedding_size": self._model.config.max_position_embeddings,
            "input_shape": {k: list(v) for k, v in model_info.input_size.items()},
            "output_shape": model_info.summary_list[-1].output_size,
            "max_context_length": self._model.config.max_length,
            "num_trainable_params": model_info.trainable_params,
            "size": model_info.total_param_bytes,
            "vocab_size": self._model.config.vocab_size
        }

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
        return self._infer_batch([sample])[0]

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

        return DataFrame({
            'target': self._dataset.data[ColumnNames.TARGET.value].tolist(),
            'predictions': predictions
        })

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        if self._model is None:
            return []

        inputs = self._tokenizer(
            sample_batch[0],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length
        ).to(self._device)

        outputs = self._model(**inputs)

        return list(str(prediction) for prediction in torch.argmax(
            outputs.logits.cpu().detach().numpy(), dim=1))


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
        self._data_path = data_path
        self._metrics = metrics

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        predictions_df = read_csv(self._data_path)
        references = predictions_df['target'].tolist()
        predictions = predictions_df['predictions'].tolist()

        results = {}
        for metric in self._metrics:
            metric_instance = load(metric.value)
            result = metric_instance.compute(predictions=predictions, references=references)
            results[metric.value] = result[metric.value]

        return results
