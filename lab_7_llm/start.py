"""
Neural machine translation starter.
"""
import json
from config.constants import PROJECT_ROOT
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main():
    """
    Run the translation pipeline.
    """

    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)
    file_importer = RawDataImporter(settings['parameters']['dataset'])
    file_importer.obtain()

    preprocessor = RawDataPreprocessor(file_importer.raw_data)

    analysis = preprocessor.analyze()
    transform = preprocessor.transform()

    result = analysis, transform

    assert result is not None, "Demo does not work correctly"

    return result


if __name__ == "__main__":
    main()
