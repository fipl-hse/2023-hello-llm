"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json
from pathlib import Path

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                           TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as settings_file:
        settings = json.load(settings_file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    assert importer.raw_data is not None, "Demo does not work correctly"
    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 120, 64, 'cpu')
    print(pipeline.analyze_model())
    infer_sample_result = pipeline.infer_sample(dataset[0])
    print(f'INPUT TEXT: {dataset[0]}\nTRANSLATION: {infer_sample_result}')

    predictions_path = Path('predictions.csv')
    pipeline.infer_dataset().to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(predictions_path, settings['parameters']['metrics'])
    result = evaluator.run()
    print(result)
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
