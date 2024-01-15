"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor
from config.constants import PROJECT_ROOT
import json
from pathlib import Path

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', "r")
    settings = json.load(settings_path)
    ds = RawDataImporter(settings["parameters"]["dataset"])
    ds.obtain()
    result = RawDataPreprocessor(ds.raw_data)
    result.analyze()
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
