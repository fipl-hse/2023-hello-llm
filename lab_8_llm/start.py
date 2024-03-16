"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_8_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        config = json.load(file)
    loader = RawDataImporter(config['parameters']['dataset'])
    loader.obtain()

    preprocessor = RawDataPreprocessor(loader.raw_data)

    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(model_name=config['parameters']['model'],
                           dataset=dataset,
                           max_length=120,
                           batch_size=1,
                           device='cpu')

    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    result = pipeline.infer_sample(dataset[0])
    print(result)
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
