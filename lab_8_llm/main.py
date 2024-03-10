"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called, duplicate-code
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchinfo import summary
from transformers import AutoModelForSequenceClassification, BertTokenizerFast

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
        self._raw_data = load_dataset(self._hf_name, split="validation").to_pandas()

        if not isinstance(self._raw_data, DataFrame):
            raise TypeError


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """
    _raw_data: DataFrame

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """

        dataset_analysis = {
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isnull().any(axis=1).sum(),
            "dataset_sample_min_len": len(min(self._raw_data["content"], key=len)),
            "dataset_sample_max_len": len(max(self._raw_data["content"], key=len))
        }
        return dataset_analysis

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """

        self._data = (
            self._raw_data[["content", "sentiment"]]
                .rename(
                columns={"sentiment": ColumnNames.TARGET.value,
                         "content": ColumnNames.SOURCE.value})
                .dropna()
                .replace({"positive": 1,
                          "negative": 2}))


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

        source = self._data[ColumnNames.SOURCE.value].iloc[index]
        target = self._data[ColumnNames.TARGET.value].iloc[index]

        return str(source), str(target)

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

        self._model_name = model_name
        self._dataset = dataset
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """

        tensor_data = torch.ones(1,
                                 self._model.config.max_position_embeddings,
                                 dtype=torch.long)

        input_data = {
            "attention_mask": tensor_data,
            "input_ids": tensor_data
        }

        model_summary = summary(model=self._model,
                                input_data=input_data,
                                device=self._device,
                                verbose=False)

        model_info = {
            "input_shape": {'attention_mask': list(model_summary.input_size["attention_mask"]),
                            'input_ids': list(model_summary.input_size["input_ids"])},
            "embedding_size": self._model.config.max_position_embeddings,
            "output_shape": model_summary.summary_list[-1].output_size,
            "num_trainable_params": model_summary.trainable_params,
            "vocab_size": self._model.config.vocab_size,
            "size": model_summary.total_param_bytes,
            "max_context_length": self._model.config.max_length
        }
        return model_info

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

        dataset_loader = DataLoader(self._dataset, self._batch_size)

        predictions = []

        for batch in dataset_loader:
            predictions.extend(['2' if i == '0' else i for i in self._infer_batch(batch)])

        df_predict = pd.DataFrame({
            "target": self._dataset.data[ColumnNames.TARGET.value],
            "predictions": predictions
        })

        return df_predict

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

        tokens = self._tokenizer(sample_batch[0], max_length=self._max_length, padding=True,
                                 return_tensors='pt', truncation=True)
        output = self._model(**tokens).logits
        predictions.extend([str(prediction.item())
                            for prediction in list(torch.argmax(output, dim=1))])
        #predictions = ['2' if i == '0' else i for i in predictions]
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
        self._data_path = data_path
        self._metrics = metrics

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """

        predictions = pd.read_csv(self._data_path)
        scores = {}

        for metric in self._metrics:
            metric = load(metric.value)

            result = metric.compute(references=predictions['target'].to_list(),
                                    predictions=predictions['predictions'].to_list(),
                                    average='micro')

            scores[metric.name] = result.get(metric.name)
        print(scores)
        return scores
