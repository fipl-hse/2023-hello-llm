"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json
from random import randint

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
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as settings_file:
        configs = json.load(settings_file)
    data_loader = RawDataImporter(configs['parameters']['dataset'])
    data_loader.obtain()

    preprocessor = RawDataPreprocessor(data_loader.raw_data)
    data_analysis = preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(configs['parameters']['model'], dataset, 120, 64, 'cpu')

    model_analysis = pipeline.analyze_model()

    sample = dataset[randint(0, len(dataset) - 1)]
    sample_inference = pipeline.infer_sample(sample)

    predictions = pipeline.infer_dataset()
    data_path = PROJECT_ROOT / 'lab_7_llm' / 'dist' / 'predictions.csv'
    pipeline.save_results(predictions, data_path)

    evaluator = TaskEvaluator(
        data_path,
        [Metrics[metric.upper()] for metric in configs['parameters']['metrics']]
    )
    result = evaluator.run()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
