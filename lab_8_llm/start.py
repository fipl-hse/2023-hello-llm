"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
import json

from config.constants import PROJECT_ROOT
from lab_8_llm.main import (
    RawDataImporter,
    RawDataPreprocessor
)

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    path_settings = PROJECT_ROOT / "lab_8_llm" / "settings.json"

    with open(path_settings, encoding="utf-8") as path:
        settings = json.load(path)
    data_loader = RawDataImporter(settings["parameters"]["dataset"])
    data_loader.obtain()

    preprocessor = RawDataPreprocessor(data_loader.raw_data)
    analysis = preprocessor.analyze()
    print(analysis)
    preprocessor.transform()

    result = analysis
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
