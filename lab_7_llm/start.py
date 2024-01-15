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
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as settings_file:
        configs = json.load(settings_file)
    data_loader = RawDataImporter(configs['parameters']['dataset'])
    data_loader.obtain()

    preprocessor = RawDataPreprocessor(data_loader.raw_data)
    result = preprocessor.analyze()
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
