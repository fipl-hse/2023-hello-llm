"""
Settings manager.
"""

# pylint: disable=no-name-in-module
from pathlib import Path

from pydantic.dataclasses import dataclass

from core_utils.llm.metrics import Metrics


@dataclass
class ParametersModel:
    """
    Additional parameters of a lab.
    """
    model: str
    dataset: str
    metrics: list[Metrics]


@dataclass
class LabSettingsModel:
    """
    DTO for storing labs settings.
    """
    parameters: ParametersModel | None
    target_score: int


class LabSettings:
    """
    Main model for working with settings.
    """
    # Labs settings
    _dto: LabSettingsModel

    def __init__(self, config_path: Path) -> None:
        """
        Initialize LabSettings.

        Args:
            config_path (Path): Path to configuration
        """
        super().__init__()
        with config_path.open(encoding='utf-8') as config_file:
            # pylint: disable=no-member
            self._dto = LabSettingsModel.__pydantic_validator__.validate_json(config_file.read())

    @property
    def target_score(self) -> int:
        """
        Property for target score.

        Returns:
            int: A target score.
        """
        return self._dto.target_score

    @property
    def parameters(self) -> ParametersModel | None:
        """
        Property for additional parameters.

        Returns:
            ParametersModel | None: Parameters DTO.
        """
        return self._dto.parameters
