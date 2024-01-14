"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor
import json
from pathlib import Path

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = Path("settings.json")
    if settings_path.is_file():
        settings = open("settings.json", "r")
    else:
        settings = open("../settings.json", "r")
    settings = json.load(settings)
    result = RawDataImporter(settings["parameters"]["dataset"])
    result.obtain()
    result = RawDataPreprocessor(result.raw_data())
    result.analyze()
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
