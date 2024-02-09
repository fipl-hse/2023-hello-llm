"""
Neural summarization starter.
"""
import json
import os
from pathlib import Path

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the summarization pipeline.
    """
    with open(PROJECT_ROOT / "lab_7_llm" / "settings.json", "r", encoding="utf-8") as settings_file:
        settings = json.load(settings_file)

    importer = RawDataImporter(hf_name=settings["parameters"]["dataset"])
    importer.obtain()

    preprocessor = RawDataPreprocessor(raw_data=importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(model_name=settings["parameters"]["model"],
                           dataset=dataset,
                           max_length=120,
                           batch_size=64,
                           device="cpu")

    if not os.path.exists(f'{PROJECT_ROOT}/lab_7_llm/dist/predictions.csv'):
        os.mkdir(f'{PROJECT_ROOT}/lab_7_llm/dist')
    pipeline.infer_dataset().to_csv(f'{PROJECT_ROOT}/lab_7_llm/dist/predictions.csv', index=False)

    result = TaskEvaluator(data_path=Path(f'{PROJECT_ROOT}/lab_7_llm/dist/predictions.csv'),
                           metrics=settings["parameters"]["metrics"])

    result.run()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
