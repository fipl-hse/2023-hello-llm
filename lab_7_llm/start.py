"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from main import RawDataImporter

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as set_file:
        configs = json.load(set_file)

    dataset = RawDataImporter(configs['parameters']['dataset'])
    dataset.obtain()

    result = None
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
