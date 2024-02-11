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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
        self._raw_data = load_dataset(self._hf_name, name='default', split='test').to_pandas()

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
        analysis = {
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': len(self._raw_data[self._raw_data.duplicated()]),
            'dataset_empty_rows': len(self._raw_data[self._raw_data.isna().any(axis=1)]),
            'dataset_sample_min_len': len(min(self._raw_data["info"], key=len)),
            'dataset_sample_max_len': len(max(self._raw_data["info"], key=len))
        }
        return analysis

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
#        self._data = (self._raw_data
#                      .rename(columns={'info': 'source', 'summary': 'target'})
#                      .reset_index(drop=True))
        self._data = self._raw_data.rename(columns={
            'info': ColumnNames.SOURCE,
            'summary': ColumnNames.TARGET}).reset_index(drop=True)


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
        return self._data.iloc[index][ColumnNames.SOURCE],
 #       return self._data['source'].iloc[index]
#        return self._data.iloc[index]['source']

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
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        config = self._model.config
        embeddings_length = config.decoder.max_position_embeddings
        ids = torch.ones(self._batch_size, embeddings_length, dtype=torch.long)
        input_data = {'input_ids': ids,
                      'attention_mask': ids,
                      'decoder_input_ids': ids}
        model_summary = summary(self._model, input_data=input_data, verbose=False)

        analysis = {
            'input_shape': list(ids.shape),
            'embedding_size': embeddings_length,
            'output_shape': model_summary.summary_list[-1].output_size,
            'num_trainable_params': model_summary.trainable_params,
            'vocab_size': config.decoder.vocab_size,
            'size': model_summary.total_param_bytes,
            'max_context_length': config.max_length
        }
        return analysis

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
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        tokens = tokenizer(sample, return_tensors='pt')
        output = self._model.generate(**tokens)
        results = tokenizer.batch_decode(output, skip_special_tokens=True)
        return results

    @report_time
    def infer_dataset(self) -> DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        data_loader = DataLoader(dataset=self._dataset,
                                 batch_size=self._batch_size)
        predictions = []
        for batch in data_loader:
            batch_predictions = self._infer_batch(batch)
            predictions.extend(batch_predictions)

        predictions_df = DataFrame({
            "target": self._dataset.data[ColumnNames.TARGET],
            "predictions": predictions
        })

        return predictions_df

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
        predictions = []

        for index, sample in enumerate(sample_batch[0]):
            tokens = tokenizer(sample_batch[0][index],
                               max_length=120,
                               padding=True,
                               return_tensors='pt',
                               truncation=True)
            output = self._model.generate(**tokens)
            result = tokenizer.batch_decode(output,
                                            skip_special_tokens=True)
            predictions.extend(result)

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

        predictions = pd.read_csv(self._data_path)
        acc_metric = [metric.value for metric in self._metrics if metric.value == 'accuracy'][0]
        metric = load(acc_metric)
        result = metric.compute(references=predictions['target'],
                                predictions=predictions['predictions'])
        return dict(result)
