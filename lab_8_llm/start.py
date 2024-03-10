"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    settings = LabSettings(PROJECT_ROOT / 'lab_8_llm' / 'settings.json')

    dataset = RawDataImporter(settings.parameters.dataset)
    dataset.obtain()
    if dataset.raw_data is None:
        raise TypeError

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

    analysis = pipe.analyze_model()
    print(analysis)
    print(pipe.infer_sample(task_dataset[0]))

    predictions = pipe.infer_dataset()
    path = PROJECT_ROOT / 'lab_8_llm' / 'dist'
    if not path.exists():
        path.mkdir()
    predictions.to_csv(path / 'predictions.csv')

    result = TaskEvaluator(path / 'predictions.csv', Metrics).run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
