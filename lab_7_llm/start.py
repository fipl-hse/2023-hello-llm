"""
Neural machine translation starter.
"""
# pylint: disable=too-many-locals
import json
import os
from pathlib import Path

from config.constants import PROJECT_ROOT
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as settings:
        settings = json.load(settings)
    raw_data_importer = RawDataImporter(settings["parameters"]["dataset"])
    raw_data_importer.obtain()
    preprocessor = RawDataPreprocessor(raw_data_importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(10))
    pipeline = LLMPipeline(settings['parameters']['model'],
                           dataset,
                           max_length=120,
                           batch_size=1,
                           device='cpu')

    pipeline.analyze_model()
    pipeline.infer_sample(next(iter(dataset)))

    if not os.path.exists(f'{PROJECT_ROOT}/lab_7_llm/dist'):
        os.mkdir(f'{PROJECT_ROOT}/lab_7_llm/dist')
    pred_path = f'{PROJECT_ROOT}/lab_7_llm/dist/predictions.csv'

    pipeline2 = LLMPipeline(settings["parameters"]["model"],
                            dataset,
                            max_length=120,
                            batch_size=64,
                            device="cpu")

    pipeline2.infer_dataset().to_csv(pred_path, index=False)

    evaluator = TaskEvaluator(Path(pred_path),
                              [Metrics[metric.upper()] for metric in
                                       settings['parameters']['metrics']])

    result = evaluator.run()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
