"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from main import RawDataImporter, RawDataPreprocessor

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    dataset = RawDataImporter('self._hf_data')
    dataset.obtain()
    with open('settings.json') as settings_file:
        with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as file:
            settings = json.load(file)
        importer = RawDataImporter(settings['parameters']['dataset'])
        importer.obtain()

        preprocessor = RawDataPreprocessor(importer.raw_data)
        result = preprocessor.analyze()
        assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
