"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
<<<<<<< HEAD
import json

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor
=======
from core_utils.llm.time_decorator import report_time
>>>>>>> ddb96a0fabdf4786c4f85f952fbba93c1002219a


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
<<<<<<< HEAD
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)
    data_importer = RawDataImporter(settings['parameters']['dataset'])
    data_importer.obtain()

    preprocessor = RawDataPreprocessor(data_importer.raw_data)
    result = preprocessor.analyze()

=======
    result = None
>>>>>>> ddb96a0fabdf4786c4f85f952fbba93c1002219a
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
