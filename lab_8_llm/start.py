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
from lab_8_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_8_llm' / 'settings.json', 'r', encoding='utf-8') as settings_file:
        settings = json.load(settings_file)
    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(5))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset,
                           max_length=120, batch_size=2, device='cpu')

    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    result = pipeline.infer_sample(dataset[0])
    print(dataset[0])
    print(result)

    if not os.path.exists(PROJECT_ROOT / 'lab_8_llm' / 'dist'):
        os.mkdir(PROJECT_ROOT / 'lab_8_llm' / 'dist')

    prediction_path = PROJECT_ROOT / 'lab_8_llm' / 'dist' / 'predictions.csv'

    pipeline.infer_dataset().to_csv(prediction_path, index=False)

    metrics = [Metrics[metric.upper()] for metric in settings['parameters']['metrics']]

    evaluator = TaskEvaluator(Path(prediction_path), metrics)

    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
