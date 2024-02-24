"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time

from lab_8_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_8_llm" / 'settings.json')

    ds_obtainer = RawDataImporter(settings.parameters.dataset)
    ds_obtainer.obtain()

    result = RawDataPreprocessor(ds_obtainer.raw_data)
    print(result.analyze())
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
