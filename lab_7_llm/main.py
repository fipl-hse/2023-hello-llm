"""
Neural machine translation module.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
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
        raw_dataset = load_dataset("RussianNLP/russian_super_glue",
                                   name=self._hf_name,
                                   split='validation')
        self._raw_data = pd.DataFrame(raw_dataset)

    @property
    def raw_data(self) -> DataFrame:
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
        analyze_dict = {
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": len(self._raw_data[self._raw_data.duplicated()]),
            "dataset_empty_rows": len(self._raw_data[self._raw_data.isna().any(axis=1)]),
            "dataset_sample_min_len": min(self._raw_data['premise'].str.len().min(),
                                          self._raw_data['hypothesis'].str.len().min()),
            "dataset_sample_max_len": max(self._raw_data['premise'].str.len().max(),
                                          self._raw_data['hypothesis'].str.len().max())
        }
        return analyze_dict

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = (self._raw_data.rename(columns={
            "label": "target"
        }
        )
                      .drop_duplicates(subset=['premise', 'hypothesis'], keep='last')
                      .dropna()
                      .reset_index(drop=True)
                      .drop(['idx'], axis=1)
                      .replace({0: 1, 1: 0}))


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
        return self._data.shape[0]

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        return str(self._data.iloc[index].premise), str(self._data.iloc[index].hypothesis)

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
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._dataset = dataset
        self._batch_size = batch_size

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        tensor_data = torch.ones(1, self._model.config.max_position_embeddings, dtype=torch.long)
        input_data = {'attention_mask': tensor_data,
                      "input_ids": tensor_data}
        analytics = summary(self._model, input_data=input_data, verbose=False)
        return {
            "embedding_size": self._model.config.max_position_embeddings,
            "input_shape": {'attention_mask': list(analytics.input_size['attention_mask']),
                            'input_ids': list(analytics.input_size['input_ids'])},
            "max_context_length": self._model.config.max_position_embeddings,
            "num_trainable_params": analytics.trainable_params,
            "output_shape": analytics.summary_list[1].output_size,
            "size": analytics.total_param_bytes,
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
        if len(sample) < 2:
            sample_to_list = sample[0].split('|')
            sample_seq = [(sample_to_list[0],), (sample_to_list[1],)]
        else:
            sample_seq = [(sample[0],), (sample[1],)]
        prediction = self._infer_batch(sample_seq)

        return str(prediction[0])

    @report_time
    def infer_dataset(self) -> DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """

        dloader = DataLoader(dataset=self._dataset, batch_size=self._batch_size)

        ds_pred_list = []

        for batch in dloader:

            batch_predictions = self._infer_batch(batch)

            for batch_pred in batch_predictions:
                ds_pred_list.append(batch_pred)

        result_df = pd.DataFrame({
            "target": self._dataset.data['target'],
            "prediction": ds_pred_list
        })

        return result_df

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        batch_pred_list = []

        for sequence in sample_batch[0]:
            hypothesis_index = sample_batch[0].index(sequence)
            sequence_tokens = self._tokenizer(sample_batch[0][hypothesis_index],
                                              sample_batch[1][hypothesis_index],
                                              return_tensors="pt",
                                              padding=True,
                                              truncation=True
                                              )
            sequence_prediction = (torch.argmax(self._model(**sequence_tokens).logits, dim=1)
                                   .tolist())

            for pred in sequence_prediction:
                batch_pred_list.append(str(pred))

        return batch_pred_list


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
        self._metrics = metrics

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        df_to_evaluate = pd.read_csv(self._data_path)
        acc_metric = [metric.value for metric in self._metrics if metric.value == 'accuracy'][0]
        metric = load(acc_metric)
        metrics_evaluation = metric.compute(references=df_to_evaluate['target'].tolist(),
                                            predictions=df_to_evaluate['prediction'].tolist()
                                            )
        return metrics_evaluation
