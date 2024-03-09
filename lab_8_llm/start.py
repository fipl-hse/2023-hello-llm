"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json

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
    with open(PROJECT_ROOT / 'lab_8_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 512, 64, 'cpu')
    print(pipeline.analyze_model())
    print(pipeline.infer_sample(dataset[0]))

    predictions = pipeline.infer_dataset()

    save_path = PROJECT_ROOT / 'lab_8_llm' / 'dist' / 'predictions.csv'
    save_path.parent.mkdir(exist_ok=True)
    predictions.to_csv(save_path)

    evaluator = TaskEvaluator(
        save_path,
        [Metrics[metric.upper()] for metric in settings['parameters']['metrics']]
    )
    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
