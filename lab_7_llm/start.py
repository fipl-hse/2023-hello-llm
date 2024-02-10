"""
Neural summarization starter.
"""
import json

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the summarization pipeline.
    """

    # mark4
    with open(PROJECT_ROOT / "lab_7_llm" / "settings.json", "r", encoding="utf-8") as settings_file:
        settings = json.load(settings_file)

    importer = RawDataImporter(hf_name=settings["parameters"]["dataset"])
    importer.obtain()

    # mark6
    preprocessor = RawDataPreprocessor(raw_data=importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline_batch1 = LLMPipeline(model_name=settings["parameters"]["model"],
                                  dataset=dataset,
                                  max_length=120,
                                  batch_size=1,
                                  device="cpu")
    pipeline_batch1.analyze_model()
    pipeline_batch1.infer_sample(next(iter(dataset)))

    # mark 8
    predictions_path = PROJECT_ROOT / "lab_7_llm" / "dist"

    if not predictions_path.exists():
        predictions_path.mkdir()

    pipeline_batch64 = LLMPipeline(model_name=settings["parameters"]["model"],
                                   dataset=dataset,
                                   max_length=120,
                                   batch_size=64,
                                   device="cpu")

    pipeline_batch64.infer_dataset()\
        .to_csv(predictions_path / "predictions.csv", index=False, encoding="utf-8")

    result = TaskEvaluator(data_path=predictions_path / "predictions.csv",
                           metrics=settings["parameters"]["metrics"])

    print(result.run())

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
