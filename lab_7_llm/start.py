"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json
from core_utils.llm.time_decorator import report_time
from main import RawDataImporter


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open('settings.json') as json_file:
        settings = json.load(json_file)
    rawdata = RawDataImporter(settings['parameters']['dataset'])
    rawdata.obtain()


if __name__ == "__main__":
    main()
