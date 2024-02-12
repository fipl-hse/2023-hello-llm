"""
Neural machine translation module.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence

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
        self._raw_data = load_dataset(self._hf_name,
                                      split='validation').to_pandas()

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
        dataset_info = {
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isna().any(axis=1).sum(),
        }

        no_empty_rows_ds = self._raw_data.dropna(subset=["comment_text"])

        dataset_info["dataset_sample_min_len"] = len(min(no_empty_rows_ds["comment_text"], key=len))
        dataset_info["dataset_sample_max_len"] = len(max(no_empty_rows_ds["comment_text"], key=len))

        return dataset_info

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """

        self._data = (self._raw_data.rename(columns={"label": str(ColumnNames.TARGET),
                                                     "comment_text": str(ColumnNames.SOURCE)})
                      .drop('id', axis=1)
                      .reset_index(drop=True))


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
        return (self._data.iloc[index]['source'],)

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
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = BertForSequenceClassification.from_pretrained(self._model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """

        embeddings_length = self._model.config.max_position_embeddings
        ids = torch.ones(1, embeddings_length, dtype=torch.long)

        data = {
            'input_ids': ids,
            'attention_mask': ids
        }

        model_summary = torchinfo.summary(self._model, input_data=data, verbose=0)

        summary_dict = {
            "input_shape": {'attention_mask': list(model_summary.input_size['attention_mask']),
                            'input_ids': list(model_summary.input_size['input_ids'])},
            "embedding_size": embeddings_length,
            "output_shape": model_summary.summary_list[-1].output_size,
            "num_trainable_params": model_summary.trainable_params,
            "vocab_size": self._model.config.vocab_size,
            "size": model_summary.total_param_bytes,
            "max_context_length": self._model.config.max_length
        }
        return summary_dict

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
        dataset_predictions = []

        dataset_loader = DataLoader(self._dataset, batch_size=self._batch_size)

        for batch in dataset_loader:
            dataset_predictions.extend(self._infer_batch(batch))

        pd_predictions = pd.DataFrame(
            {'target': self._dataset.data[str(ColumnNames.TARGET)],
             'predictions': pd.Series(dataset_predictions)}
        )

        return pd_predictions

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        predictions = []

        for sample in sample_batch[0]:
            tokens = self._tokenizer(
                sample,
                max_length=self._max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            output = self._model(**tokens).logits
            predictions.extend([str(prediction.item())
                                for prediction in list(torch.argmax(output, dim=1))])

        return predictions


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
        to_eval_df = pd.read_csv(self._data_path)
        evaluations = {}
        for metric in self._metrics:
            metric = load(metric.value)
            evaluation = metric.compute(references=to_eval_df['target'].tolist(),
                                        predictions=to_eval_df['predictions'].tolist(),
                                        average='micro'
                                        )
            evaluations.update(dict(evaluation))

        return evaluations
