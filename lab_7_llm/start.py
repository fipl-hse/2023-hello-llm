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

    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as settings:
        settings_dict = json.load(settings)
    raw_dataset = RawDataImporter(settings_dict["parameters"]["dataset"])
    raw_dataset.obtain()
    processed_dataset = RawDataPreprocessor(raw_dataset.raw_data)
    #processed_dataset.analyze()
    #processed_dataset.transform()

    result = None
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
