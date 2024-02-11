"""
Neural machine translation starter.
"""

import json

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time

from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as settings:
        settings = json.load(settings)
    govreport = RawDataImporter('ccdv/govreport-summarization')
    govreport.obtain()
    preprocessor = RawDataPreprocessor(govreport.raw_data)
    preprocessor.transform()
    result = preprocessor.analyze()
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings['parameters']['model'],
                           dataset,
                           max_length=120,
                           batch_size=5,
                           device='cpu')

    pipeline.analyze_model()
    pipeline.infer_sample(dataset[0])

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
