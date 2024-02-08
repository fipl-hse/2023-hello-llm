"""
Neural machine translation starter.
"""
import json

# pylint: disable= too-many-locals
from config.constants import PROJECT_ROOT
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
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
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as config:
        data = json.load(config)
    dataset = RawDataImporter(data['parameters']['dataset'])
    dataset.obtain()
    data_preprocessor = RawDataPreprocessor(dataset.raw_data)
    print(data_preprocessor.analyze())
    data_preprocessor.transform()

    task_dataset = TaskDataset(data_preprocessor.data.head(100))
    pipe = LLMPipeline(
        data['parameters']['model'],
        task_dataset,
        batch_size=2,
        max_length=120,
        device='cpu')

    print(pipe.analyze_model())
    print(pipe.infer_sample(task_dataset[0]))
    predictions = pipe.infer_dataset()

    path = (PROJECT_ROOT / 'lab_7_llm' / 'dist')

    if not path.exists():
        path.mkdir()

    predictions.to_csv(path / 'predictions.csv')

    result = TaskEvaluator(path / 'predictions.csv', Metrics).run()

    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
