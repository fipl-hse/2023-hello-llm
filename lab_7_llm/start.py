"""
Neural machine translation starter.
"""
from pandas import DataFrame

# pylint: disable= too-many-locals

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)

SETTINGS = PROJECT_ROOT / "lab_7_llm" / "settings.json"
PREDICTIONS = PROJECT_ROOT / "lab_7_llm" / "dist" / "predictions.csv"


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    parameters = LabSettings(SETTINGS).parameters
    data_loader = RawDataImporter(parameters.dataset)
    data_loader.obtain()

    preprocessor = RawDataPreprocessor(data_loader.raw_data)
    if not isinstance(preprocessor, DataFrame):
        raise TypeError()
    print(preprocessor.analyze())
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(
        parameters.model,
        dataset,
        max_length=120,
        batch_size=64,
        device="cpu",
    )

    pipeline.analyze_model()
    pipeline.infer_sample(next(iter(dataset)))

    PREDICTIONS.parent.mkdir(exist_ok=True)
    pipeline.infer_dataset().to_csv(PREDICTIONS, index=False)

    evaluator = TaskEvaluator(PREDICTIONS, parameters.metrics)
    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
