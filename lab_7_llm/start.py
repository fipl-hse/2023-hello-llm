"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    preprocessor = RawDataPreprocessor(importer.raw_data)

    data_analysis = preprocessor.analyze()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 512, 1, 'cpu')
    model_analysis = pipeline.analyze_model()
    result = pipeline.infer_sample(dataset[0])

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
