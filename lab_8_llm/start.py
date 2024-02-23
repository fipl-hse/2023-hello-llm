"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json
from random import randint

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_8_llm' / 'settings.json')
    importer = RawDataImporter(settings.parameters.dataset)
    preprocessor = RawDataPreprocessor(importer.raw_data)
    dataset = TaskDataset(preprocessor.data.head(100))
    result = dataset
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
