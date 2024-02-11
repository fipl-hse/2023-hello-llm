"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json

from core_utils.llm.time_decorator import report_time

from config.constants import PROJECT_ROOT
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    SETTINGS = PROJECT_ROOT / "lab_7_llm" / "settings.json"

    with open(SETTINGS, encoding="utf-8") as path:
        settings = json.load(path)
    data_loader = RawDataImporter(settings["parameters"]["dataset"])
    data_loader.obtain()

    preprocessor = RawDataPreprocessor(data_loader.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

if __name__ == "__main__":
    main()
