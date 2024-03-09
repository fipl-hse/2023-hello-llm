"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from pathlib import Path

from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    root_path = Path(__file__).parent
    settings_path = LabSettings(root_path / "settings.json")
    predictions_path = root_path / "dist" / "predictions.csv"

    if not predictions_path.parent.exists():
        predictions_path.parent.mkdir(exist_ok=True)

    importer = RawDataImporter(settings_path.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        raise TypeError('Dataset is None')

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    batch_size = 64
    max_length = 120

    pipeline = LLMPipeline(model_name=settings_path.parameters.model,
                           dataset=dataset,
                           max_length=max_length,
                           batch_size=batch_size,
                           device="cpu")
    pipeline.analyze_model()
    pipeline.infer_sample(next(iter(dataset)))

    predictions = pipeline.infer_dataset()
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")

    evaluator = TaskEvaluator(predictions_path, settings_path.parameters.metrics)
    result = evaluator.run()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
