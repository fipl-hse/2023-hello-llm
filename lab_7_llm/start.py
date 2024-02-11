"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from config.constants import PROJECT_ROOT

from lab_7_llm.main import RawDataImporter, RawDataPreprocessor

import json

@report_time
def main():
    """
    Run the translation pipeline.
    """
    settings = json.load(open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', "r"))
    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)

    analysis = preprocessor.analyze()
    transform = preprocessor.transform()

    return analysis, transform

if __name__ == "__main__":
    main()
