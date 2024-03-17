"""
Neural machine translation starter.
"""
import json
from random import randint

from config.constants import PROJECT_ROOT
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline


@report_time
def main():
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_8_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)
    file_importer = RawDataImporter(settings['parameters']['dataset'])
    file_importer.obtain()

    preprocessor = RawDataPreprocessor(file_importer.raw_data)
    print(preprocessor.analyze())

    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    llm = LLMPipeline(settings["parameters"]["model"], dataset, 120, 64, "cpu")
    print(llm.analyze_model())

    result = llm.infer_sample(dataset[0])
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
