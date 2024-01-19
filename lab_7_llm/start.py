"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from config.constants import PROJECT_ROOT

from main import RawDataImporter, RawDataPreprocessor

import json


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = json.load(open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', "r"))
    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.raw_data)
    result = preprocessor.analyze()
    assert result is not None, "Demo does not work correctly"
    print(result)


if __name__ == "__main__":
    main()
