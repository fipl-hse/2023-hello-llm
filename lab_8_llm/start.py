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
    Run the closed qa pipeline.
    """
    with open(PROJECT_ROOT / 'lab_8_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)
    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)

    data_analysis = preprocessor.analyze()
    print(data_analysis)

    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset,
                           max_length=512, batch_size=64, device='cpu')

    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    sample_result = pipeline.infer_sample(dataset[0])
    print(sample_result)

    dataset_result = pipeline.infer_dataset()
    print(dataset_result)

    if not os.path.exists(PROJECT_ROOT / 'lab_8_llm' / 'dist'):
        os.mkdir(PROJECT_ROOT / 'lab_8_llm' / 'dist')

    predictions_path = PROJECT_ROOT / 'lab_8_llm' / 'dist' / 'predictions.csv'

    dataset_result.to_csv(predictions_path, index=False)

    metrics = [Metrics[metric.upper()] for metric in settings['parameters']['metrics']]

    evaluator = TaskEvaluator(Path(predictions_path), metrics)

    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
