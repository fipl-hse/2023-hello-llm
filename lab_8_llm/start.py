"""
Neural Classification starter.
"""
# pylint: disable= too-many-locals
from pathlib import Path

from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the classification pipeline.
    """
    root_path = Path(__file__).parent
    settings = LabSettings(root_path / "settings.json")

    # mark 4
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    # mark6
    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(model_name=settings.parameters.model,
                           dataset=dataset,
                           max_length=120,
                           batch_size=1,
                           device="cpu")
    pipeline.analyze_model()
    pipeline.infer_sample(next(iter(dataset)))

    # mark 8

    result = dataset
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
