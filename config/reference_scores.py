"""
Helper for reference results
"""
# pylint: disable=too-few-public-methods
import json
from enum import Enum
from pathlib import Path


class ReferenceAnalysisScoresType(Enum):
    """
    Type for analysis
    """

    MODEL = 'model'
    DATASET = 'dataset'
    INFERENCE = 'inference'


class ReferenceScores:
    """
    Manager of reference scores.
    """

    def __init__(self) -> None:
        """
        Initialize lab settings.
        """
        config_path = Path(__file__).parent / 'reference_scores.json'

        with config_path.open(encoding='utf-8') as config_file:
            self._dto = json.load(config_file)

    def get(self, model: str, dataset: str, metric: str) -> float:
        """
        Get reference result.

        Args:
            model (str): Model
            dataset (str): Dataset
            metric (str): Metric
        """
        return float(self._dto[model][dataset][metric])


class ReferenceAnalysisScores:
    """
    Manager of reference scores.
    """

    def __init__(self,
                 scores_type: ReferenceAnalysisScoresType = ReferenceAnalysisScoresType.DATASET
                 ) -> None:
        """
        Initialize lab settings.
        """
        config_path = Path(__file__).parent / f'reference_{scores_type.value}_analytics.json'

        with config_path.open(encoding='utf-8') as config_file:
            self._dto = json.load(config_file)

    def get(self, item: str) -> dict[str, int]:
        """
        Get reference result.

        Args:
            item (str): Dataset

        Returns:
            dict[str, int]: A list of predictions.
        """
        result: dict[str, int]
        result = self._dto[item]
        return result
