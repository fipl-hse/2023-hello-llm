"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json

from core_utils.llm.time_decorator import report_time

from config.constants import PROJECT_ROOT
from lab_7_llm.main import RawDataImporter


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    SETTINGS = PROJECT_ROOT / "lab_7_llm" / "settings.json"

    with open(SETTINGS, encoding="utf-8") as path:
        settings = json.load(path)
    data_importer = RawDataImporter(settings["parameters"]["dataset"])
    data_importer.obtain()


if __name__ == "__main__":
    main()
