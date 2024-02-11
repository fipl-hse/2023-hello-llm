"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json
import os
from pathlib import Path
from random import randint

from config.constants import PROJECT_ROOT
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline, TaskEvaluator


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
                           max_length=120, batch_size=1, device='cpu')
    pipeline.analyze_model()
    pipeline.infer_sample([dataset[randint(0, len(dataset))]])

    pred_path = Path(PROJECT_ROOT / 'lab_7_llm' / 'dist' / 'predictions.csv')

    if not pred_path.parent.exists():
        pred_path.parent.mkdir()

    pipeline.infer_dataset().to_csv(pred_path, index=False)

    evaluator = TaskEvaluator(
        pred_path,
        [Metrics[metric.upper()] for metric in settings['parameters']['metrics']]
    )
    result = evaluator.run()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
