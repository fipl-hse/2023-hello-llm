"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from main import RawDataImporter, RawDataPreprocessor
import json

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    dataset = RawDataImporter('self._hf_data')
    dataset.obtain()
    with open('settings.json') as settings_file:
        settings = json.load(settings_file)

    importer = RawDataImporter(settings["parameters"]["dataset"])
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.raw_data)
    result = preprocessor.analyze()


if __name__ == "__main__":
    main()
