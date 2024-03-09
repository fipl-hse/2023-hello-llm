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
from transformers import ElectraForQuestionAnswering, ElectraTokenizer

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
        analysis = {
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': len(self._raw_data[self._raw_data.duplicated()]),
            'dataset_empty_rows': len(self._raw_data[self._raw_data.isna().any(axis=1)]),
            'dataset_sample_min_len': len(min(self._raw_data["question"], key=len)),
            'dataset_sample_max_len': len(max(self._raw_data["note"], key=len))
        }
        return analysis

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.loc[
            self._raw_data['task'] == 'Question Answering'
            ].loc[:, ['note', 'question', 'answer']].rename(columns={
                'note': ColumnNames.CONTEXT.value,
                'answer': ColumnNames.TARGET.value
            })


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
        return (self._data.iloc[index][ColumnNames.QUESTION.value],
                self._data.iloc[index][ColumnNames.CONTEXT.value])

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
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device
        self._model = ElectraForQuestionAnswering.from_pretrained(self._model_name)
        self._tokenizer = ElectraTokenizer.from_pretrained(self._model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        config = self._model.config
        embeddings_length = config.max_position_embeddings
        ids = torch.ones(1, embeddings_length, dtype=torch.long)
        input_data = {'input_ids': ids,
                      'attention_mask': ids}
        model_summary = summary(self._model, input_data=input_data, verbose=False)

        analysis = {
            'input_shape': {'input_ids': list(ids.shape), 'attention_mask': list(ids.shape)},
            'embedding_size': embeddings_length,
            'output_shape': model_summary.summary_list[-1].output_size,
            'num_trainable_params': model_summary.trainable_params,
            'vocab_size': config.vocab_size,
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
        if not self._model:
            return None
        return self._infer_batch([(sample[0],), (sample[1],)])[0]

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
            predictions.extend(self._infer_batch(batch))

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
        answer_tokens = []

        tokens = self._tokenizer(*sample_batch,
                                 max_length=self._max_length,
                                 padding=True,
                                 return_tensors='pt',
                                 truncation=True)
        outputs = self._model(**tokens)

        for i, idx in enumerate(tokens['input_ids']):
            answer_start_index = outputs.start_logits[i].argmax()
            answer_end_index = outputs.end_logits[i].argmax()
            answer_tokens.append(idx[answer_start_index:answer_end_index + 1])

        result = self._tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
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
        self._data_path = data_path
        self._metrics = metrics

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        result = {}
        predictions = []
        references = []

        data = pd.read_csv(self._data_path)
        data = data.fillna(' ')
        for ind in data.index:
            prediction = {'prediction_text': data[ColumnNames.PREDICTION.value].iloc[ind],
                          'id': str(ind)}

            reference = {'id': str(ind),
                         'answers': {'answer_start': [ind],
                                     'text': [data[ColumnNames.TARGET.value].iloc[ind]]}}

            predictions.append(prediction)
            references.append(reference)

        for metric in self._metrics:
            score = load(metric.value).compute(
                references=references,
                predictions=predictions
            )
            result[metric.value] = score.get('f1')
        return result
