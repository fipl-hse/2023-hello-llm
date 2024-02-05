"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json
from random import randint

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


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
    print(data_analysis)

    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(configs['parameters']['model'], dataset, 120, 2, 'cpu')

    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    sample = dataset[randint(0, len(dataset))]
    sample_inference = pipeline.infer_sample(sample)
    print(f'SAMPLE: {sample}\nPREDICTION: {sample_inference}')

    result = sample_inference
    print(pipeline.infer_dataset())
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
