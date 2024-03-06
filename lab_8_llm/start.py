"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json
from pathlib import Path

from config.constants import PROJECT_ROOT
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / "lab_8_llm" / "settings.json", "r", encoding="utf-8") as settings_json:
        settings = json.load(settings_json)

    raw_data = RawDataImporter(settings["parameters"]["dataset"])
    raw_data.obtain()

    preprocessed_data = RawDataPreprocessor(raw_data.raw_data)
    preprocessed_data.transform()

    dataset = TaskDataset(preprocessed_data.data.head(100))
    llm = LLMPipeline(settings["parameters"]["model"],
                      dataset,
                      120,
                      2,
                      "cpu")
    llm.infer_sample(dataset[0])

    predictions = llm.infer_dataset()
    predictions_path = PROJECT_ROOT / 'lab_8_llm' / 'dist' / 'predictions.csv'
    if not predictions_path.parent.exists():
        predictions_path.parent.mkdir()
    predictions.to_csv(predictions_path, index_label='id')

    result = TaskEvaluator(
        Path(predictions_path),
        [Metrics[metric.upper()] for metric in settings['parameters']['metrics']]
    )
    print(result.run())

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
