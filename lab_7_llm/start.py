"""
Neural machine translation starter.
"""
import json

# pylint: disable= too-many-locals
from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


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
        batch_size=1,
        max_length=120,
        device='cpu')
    print(pipe.analyze_model())
    result = pipe.infer_sample(task_dataset[0])
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
