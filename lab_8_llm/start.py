"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    settings = LabSettings(PROJECT_ROOT / 'lab_8_llm' / 'settings.json')

    dataset = RawDataImporter(settings.parameters.dataset)
    dataset.obtain()

    data_preprocessor = RawDataPreprocessor(dataset.raw_data)
    print(data_preprocessor.analyze())
    data_preprocessor.transform()

    task_dataset = TaskDataset(data_preprocessor.data.head(100))
    pipe = LLMPipeline(
        settings.parameters.model,
        task_dataset,
        batch_size=64,
        max_length=120,
        device='cpu')
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
