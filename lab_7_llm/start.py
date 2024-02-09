"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from main import RawDataImporter
import json
from pathlib import Path

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    with open(Path('settings.json'), 'r', encoding='utf-8') as settings_file:
        settings = json.load(settings_file)
    dataset = RawDataImporter(settings['parameters']['dataset'])
    dataset.obtain()


if __name__ == "__main__":
    main()
