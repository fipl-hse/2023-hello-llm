"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from pathlib import Path

from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the summarization pipeline.
    """

    root_path = Path(__file__).parent
    settings = LabSettings(root_path / "settings.json")

    # mark 4
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    result = dataset
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
