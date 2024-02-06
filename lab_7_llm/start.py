"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json
from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, LLMPipeline, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)
    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(model_name=settings['parameters']['model'], dataset=dataset,
                           max_length=120,
                           batch_size=1,
                           device='cpu')

    analysis = pipeline.analyze_model()
    # inference = pipeline.infer_sample()

    assert analysis is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
