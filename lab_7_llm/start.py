"""
Neural machine translation starter.
"""
import json
from pathlib import Path
from pprint import pprint

from config.constants import PROJECT_ROOT
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
# pylint: disable= too-many-locals
from lab_7_llm.main import (LLMPipeline,
                            RawDataImporter,
                            RawDataPreprocessor,
                            TaskDataset,
                            TaskEvaluator)

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None

    predictions_path = f'{PROJECT_ROOT}/lab_7_llm/results_csv/predictions.csv'

    with open(PROJECT_ROOT / "lab_7_llm" / "settings.json", "r", encoding="utf-8") as settings_json:
        settings = json.load(settings_json)

    raw_data = RawDataImporter(settings["parameters"]["dataset"])
    raw_data.obtain()

    preprocessed_data = RawDataPreprocessor(raw_data.raw_data)
    pprint(preprocessed_data.analyze())

    preprocessed_data.transform()

    dataset = TaskDataset(preprocessed_data.data.head(100))

    llm = LLMPipeline(settings["parameters"]["model"], dataset, 120, 2, "cpu")

    pprint(llm.analyze_model())

    sample_infer = llm.infer_sample(dataset[0])

    print('prediction for sample (',dataset[0],')',sample_infer)

    predictions = llm.infer_dataset().to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(data_path=Path(predictions_path), metrics=Metrics)
    evaluator.run()

    result = evaluator

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
