"""
Neural machine translation starter.
"""
import json

# pylint: disable= too-many-locals
from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as config:
        data = json.load(config)
    dataset = RawDataImporter(data['parameters']['dataset'])
    dataset.obtain()
    result = RawDataPreprocessor(dataset.raw_data).analyze()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
