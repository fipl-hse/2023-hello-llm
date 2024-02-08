"""
Neural machine translation module.
"""
# pylint: disable=too-few-public-methods,
# undefined-variable, too-many-arguments,
# super-init-not-called
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from datasets import load_dataset
from evaluate import load
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import AutoTokenizer, BertForSequenceClassification

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
    A class that imports the HuggingFace dataset.
    """

    def __init__(self, hf_name: str | None):
        super().__init__(hf_name)

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        self._raw_data = (load_dataset(self._hf_name, split='dev')
                          .filter(lambda dataset: dataset['source'] == 'mnli')
                          .to_pandas())


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
        return {'dataset_number_of_samples': self._raw_data.shape[0],
                'dataset_columns': self._raw_data.shape[1],
                'dataset_duplicates': self._raw_data.duplicated().sum(),
                'dataset_empty_rows': self._raw_data.isna().sum().sum(),
                'dataset_sample_min_len': min(len(min(self._raw_data["premise_ru"], key=len)),
                                              len(min(self._raw_data["hypothesis_ru"], key=len))),
                'dataset_sample_max_len': max(len(max(self._raw_data["premise_ru"], key=len)),
                                              len(max(self._raw_data["hypothesis_ru"], key=len)))
                }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = (self._raw_data.
                      loc[self._raw_data["source"] == "mnli"].
                      loc[:, ["premise_ru", "hypothesis_ru", "label"]].
                      rename(columns={
                                "premise_ru": "premise",
                                "hypothesis_ru": "hypothesis",
                                "label": "target"}).
                      dropna().
                      reset_index())

        self._data["target"].replace({"contradiction": 1,
                                      "entailment": 0,
                                      "neutral": 2}, inplace=True)


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
        return self._data.iloc[index]["premise"], self._data.iloc[index]["hypothesis"]

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

    def __init__(self, model_name: str, dataset: TaskDataset, max_length: int, batch_size: int,
                 device: str) -> None:
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
        tensor_data = torch.ones(1,
                                 self._model.config.max_position_embeddings,
                                 dtype=torch.long)
        input_data = {"input_ids": tensor_data,
                      "attention_mask": tensor_data}
        model_statistics = summary(self._model,
                                   input_data=input_data,
                                   verbose=False)
        size, num_trainable_params, last_layer = (model_statistics.total_params,
                                                  model_statistics.trainable_params,
                                                  model_statistics.summary_list[-1].output_size)
        return {"input_shape": input_data,
                "embedding_size": self._model.config.max_position_embeddings,
                "output_shape": last_layer,
                "num_trainable_params": num_trainable_params,
                "vocab_size": self._model.config.vocab_size,
                "size": size,
                "max_context_length": self._model.config.max_position_embeddings}

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        return self._infer_batch((sample,))[0]

    @report_time
    def infer_dataset(self) -> DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        loader = DataLoader(self._dataset, batch_size=self._batch_size)
        prediction = []

        for batch in loader:
            prediction.extend(self._infer_batch(batch))

        prediction = pd.Series(prediction, name="predictions")

        return pd.concat([self._dataset.data["target"], prediction], axis=1)

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

        if isinstance(sample_batch, tuple):
            sample_batch = sample_batch[0]

        tokens = tokenizer(sample_batch[0],
                           sample_batch[1],
                           padding=True,
                           truncation=True,
                           return_tensors='pt')

        output = self._model(**tokens)
        predictions = torch.argmax(output.logits, dim=1)

        predictions = [str(prediction.item()) for prediction in predictions]
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
        data = pd.read_csv(self._data_path).drop(columns=['Unnamed: 0'])

        for metric in self._metrics:
            if metric.value == "accuracy":
                accuracy_metric = load(metric.value).compute(references=data['target'],
                                                             predictions=data['predictions'])
                return accuracy_metric
