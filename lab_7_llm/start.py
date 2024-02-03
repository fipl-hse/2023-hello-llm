"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json

from core_utils.llm.time_decorator import report_time
from config.constants import PROJECT_ROOT

from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as settings_file:
        settings = json.load(settings_file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 120, 1, 'cpu')
    print(pipeline.analyze_model())
    print(f'INPUT TEXT: {dataset[0][0]}\nTRANSLATION: {pipeline.infer_sample(dataset[0])}')


if __name__ == "__main__":
    main()
