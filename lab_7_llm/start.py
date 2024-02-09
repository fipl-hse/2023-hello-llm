"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals

import json
import os
from pathlib import Path

from config.constants import PROJECT_ROOT
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset, TaskEvaluator


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as settings:
        settings = json.load(settings)
    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset,
                           max_length=120, batch_size=64, device='cpu')

    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    pipeline.infer_sample(dataset[0])

    if not os.path.exists(PROJECT_ROOT / 'lab_7_llm' / 'dist'):
        os.mkdir(PROJECT_ROOT / 'lab_7_llm' / 'dist')

    prediction_path = PROJECT_ROOT / 'lab_7_llm' / 'dist' / 'predictions.csv'

    pipeline.infer_dataset().to_csv(prediction_path, index=False)

    evaluator = TaskEvaluator(Path(prediction_path),
                              [Metrics[metric.upper()] for metric in settings['parameters']['metrics']])

    result = evaluator.run()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
