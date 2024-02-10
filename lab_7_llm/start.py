"""
Neural machine translation starter.
"""
from pathlib import Path

from config.lab_settings import LabSettings
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    root_path = Path(__file__).parent
    settings = LabSettings(root_path / "settings.json")
    predictions_path = root_path / "dist" / "predictions.csv"

    if not predictions_path.parent.exists():
        predictions_path.parent.mkdir(exist_ok=True)

    device = "cpu"
    batch_size = 1
    max_length = 120
    num_samples = 10

    raw_data_importer = RawDataImporter(settings.parameters.dataset)
    raw_data_importer.obtain()

    if raw_data_importer.raw_data is None:
        raise ValueError("raw_data_importer.raw_data is None.")

    raw_data_preprocessor = RawDataPreprocessor(raw_data_importer.raw_data)
    print(raw_data_preprocessor.analyze())

    # Mark 6
    raw_data_preprocessor.transform()
    dataset = TaskDataset(raw_data_preprocessor.data.head(num_samples))
    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)

    analysis = pipeline.analyze_model()
    print(analysis)

    prediction = pipeline.infer_sample(next(iter(dataset)))
    print(prediction)

    # Mark 8
    batch_size = 64
    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)
    predictions_df = pipeline.infer_dataset()
    predictions_df.to_csv(predictions_path, index=False, encoding='utf-8')

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
