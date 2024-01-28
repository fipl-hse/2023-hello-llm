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
    # result = None
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        config = json.load(file)
    loader = RawDataImporter(config['parameters']['dataset'])
    loader.obtain()

    preprocessor = RawDataPreprocessor(loader.raw_data)
    result = preprocessor.analyze()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
