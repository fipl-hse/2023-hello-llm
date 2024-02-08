"""
Neural machine translation module.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torchinfo
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification

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
    _raw_data: DataFrame

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        self._raw_data = load_dataset(self._hf_name, 'ru', split='validation').to_pandas()

    @property
    def raw_data(self) -> DataFrame:
        """
        Property for original dataset in a table format.

        Returns:
            pandas.DataFrame: A dataset in a table format
        """
        return self._raw_data


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
        info = {
            'dataset_number_of_samples': self._count_samples(),
            'dataset_columns': self._count_columns(),
            'dataset_duplicates': self._count_duplicates(),
            'dataset_empty_rows': self._count_empty(),
            'dataset_sample_min_len': self._count_min(),
            'dataset_sample_max_len': self._count_max()
            }
        return info

    def _count_samples(self) -> int:
        """
        Count number of rows in a DataFrame.

        Returns:
            int: number of rows in a DataFrame
        """
        return len(self._raw_data)

    def _count_columns(self) -> int:
        """
        Count number of columns in a DataFrame.

        Returns:
            int: number of columns in a DataFrame
        """
        return self._raw_data.shape[1]

    def _count_duplicates(self) -> int:
        """
        Count number of duplicates in a DataFrame.

        Returns:
            int: number of duplicates in a DataFrame
        """
        return sum(self._raw_data.duplicated())

    def _count_empty(self) -> int:
        """
        Count number of empty rows in a DataFrame including those having empty strings.

        Returns:
            int: number of empty rows in a DataFrame
        """
        return len(self._raw_data) - len(self._raw_data.replace('', np.nan).dropna())

    def _count_min(self) -> int:
        """
        Count length of the shortest sample.

        Returns:
            int: length of the shortest sample
        """
        return min(len(min(self._raw_data['premise'], key=len)),
                   len(min(self._raw_data['hypothesis'], key=len)))

    def _count_max(self) -> int:
        """
        Count length of the longest sample.

        Returns:
            int: length of the longest sample
        """
        return max(len(max(self._raw_data['premise'], key=len)),
                   len(max(self._raw_data['hypothesis'], key=len)))

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = (self._raw_data
                      .rename(columns={'label': ColumnNames['TARGET'].value})
                      .drop_duplicates()
                      .replace('', np.nan).dropna()
                      .reset_index(drop=True)
                      )


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
        super().__init__()
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
        return self._data.iloc[index]['premise'], self._data.iloc[index]['hypothesis']

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
    _model: torch.nn.Module

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
        super().__init__(model_name, dataset, max_length, batch_size, device)
        self._model = BertForSequenceClassification.from_pretrained(self._model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        embeddings_length = self._model.config.max_position_embeddings
        ids = torch.ones(self._batch_size, embeddings_length, dtype=torch.long)
        model_summary = self._get_summary(ids)
        input_shape = {
            'input_ids': [ids.shape[0], ids.shape[1]],
            'attention_mask': [ids.shape[0], ids.shape[1]]
        }

        info = {
            'input_shape': input_shape,
            'embedding_size': embeddings_length,
            'output_shape': model_summary.summary_list[-1].output_size,
            'num_trainable_params': model_summary.trainable_params,
            'vocab_size': self._model.config.vocab_size,
            'size': model_summary.total_param_bytes,
            'max_context_length': 'idk where to get it'
        }
        return info

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        return None if self._model is None else self._infer_batch((sample,))[0]

    @report_time
    def infer_dataset(self) -> DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        predictions = []
        loader = DataLoader(self._dataset, batch_size=self._batch_size)
        for batch in loader:
            predictions.extend(self._infer_batch(batch))
        return pd.DataFrame(
            pd.concat(
                [self._dataset.data['target'], pd.Series(predictions, name='predictions')],
                axis=1
            )
        )

    def _get_summary(self, ids: torch.Tensor) -> torchinfo.model_statistics.ModelStatistics:
        """
        Get model summary using torchinfo.

        Args:
            ids (torch.Tensor): input data imitation

        Returns:
            torchinfo.model_statistics.ModelStatistics: model summary
        """
        data = {
            'input_ids': ids,
            'attention_mask': ids
        }
        return torchinfo.summary(self._model, input_data=data, verbose=0)

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if len(sample_batch) == 1:
            tokens = tokenizer(
                sample_batch[0][0],
                sample_batch[0][1],
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
        else:
            tokens = tokenizer(
                sample_batch[0],
                sample_batch[1],
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
        output = self._model(**tokens).logits
        return [str(prediction.item()) for prediction in list(torch.argmax(output, dim=1))]

    @staticmethod
    def save_results(predictions: pd.DataFrame, save_path: Path) -> None:
        """
        Save dataframe with predictions as csv file.
        This will rewrite existing files in destination folder.

        Args:
            predictions (pd.DataFrame): Dataframe with model's predictions
            save_path (Path): Path to save predictions
        """
        if not save_path.parent.exists():
            save_path.parent.mkdir()
        predictions.to_csv(save_path, index_label='id')


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
        super().__init__(metrics)
        self._data_path = data_path

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        scores = {}
        for metric in self._metrics:
            evaluator = load(metric.value)
            predictions = self._load_data()
            score = evaluator.compute(
                predictions=predictions['predictions'],
                references=predictions['target']
            )
            scores.update(score)
        return scores

    def _load_data(self) -> pd.DataFrame:
        """
        Load predictions data

        Returns:
            pd.DataFrame: A dataframe with predictions and actual labels
        """
        return pd.read_csv(self._data_path, index_col='id')
