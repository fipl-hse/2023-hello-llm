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
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

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
        dataset = load_dataset(self._hf_name, split="test")
        self._raw_data = dataset.to_pandas()

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError


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
        analyzed = {'dataset_number_of_samples': len(self._raw_data),
                    'dataset_columns': len(self._raw_data.columns),
                    'dataset_duplicates': self._raw_data.duplicated().sum(),
                    'dataset_empty_rows': self._raw_data.isna().sum().sum(),
                    'dataset_sample_min_len': len(min(self._raw_data['instruction'], key=len)),
                    'dataset_sample_max_len': len(max(self._raw_data['context'], key=len))}

        return analyzed

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = (self._raw_data[['instruction', 'context', 'response']]
                      .rename(columns={'instruction': ColumnNames.QUESTION.value,
                                       'response': ColumnNames.TARGET.value})
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
        return self._data['question'].iloc[index], self._data['context'].iloc[index]

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
        self._model = AutoModelForQuestionAnswering.from_pretrained(self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        embeddings_length = self._model.config.max_position_embeddings
        tensor = torch.ones(1, embeddings_length, dtype=torch.long)

        ids = {"input_ids": tensor,
               "attention_mask": tensor}

        statistics = summary(self._model,
                             input_data=ids,
                             verbose=False)

        input_size = {"attention_mask": list(statistics.input_size['attention_mask']),
                      "input_ids": list(statistics.input_size['input_ids'])}

        model_info = {
            "input_shape": input_size,
            "embedding_size": embeddings_length,
            "output_shape": statistics.summary_list[-1].output_size,
            "num_trainable_params": statistics.trainable_params,
            "vocab_size": self._model.config.vocab_size,
            "size": statistics.total_param_bytes,
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
        all_predictions = []
        for batch in dataset_loader:
            all_predictions.extend(self._infer_batch(batch))
        df_predict = pd.DataFrame({
            "target": self._dataset.data['target'],
            "predictions": all_predictions
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
        predict_answer_tokens = []

        tokens = self._tokenizer(sample_batch[0], sample_batch[1], max_length=512, padding=True,
                                 return_tensors='pt', truncation=True)
        outputs = self._model(**tokens)

        for i, idx in enumerate(tokens['input_ids']):
            answer_start_index = outputs.start_logits[i].argmax()
            answer_end_index = outputs.end_logits[i].argmax()
            predict_answer_tokens.append(idx[answer_start_index:answer_end_index + 1])

        result = self._tokenizer.batch_decode(predict_answer_tokens, skip_special_tokens=True)
        predictions.extend(result)

        return predictions


def convert_to_squad(data: dict) -> tuple[list[dict], list[dict]]:
    """
    Convert the data into a special structure for squad metric.

    Args:
        data (dict): Data with predictions
    Returns:
        tuple[list[dict], list[dict]]: Lists of dictionaries
    """
    list_for_squad_r = []
    list_for_squad_p = []

    for i in data['index']:
        reference = {'predictions': {}, 'references': {}}

        reference['predictions']['id'] = str(i)
        reference['references']['id'] = str(i)

        reference['predictions']['prediction_text'] = data['data'][i][1]

        reference['references']['answers'] = {}
        reference['references']['answers']['text'] = [data['data'][i][0]]
        reference['references']['answers']['answer_start'] = [i]

        list_for_squad_r.append(reference['references'])
        list_for_squad_p.append(reference['predictions'])

    return list_for_squad_r, list_for_squad_p


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
        scores = {}
        data = predictions.to_dict('split')

        data_for_squad = convert_to_squad(data)

        for metric in self._metrics:
            metric = load(metric.value)
            result = metric.compute(references=data_for_squad[0], predictions=data_for_squad[1])
            scores[metric.name] = result['f1']

        return scores
