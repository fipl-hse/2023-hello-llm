"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from main import RawDataImporter, RawDataPreprocessor
import json
from config.constants import PROJECT_ROOT
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torchinfo import summary


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open('settings.json', 'r', encoding='utf-8') as settings:
        settings = json.load(settings)
    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    dataset_analysis = preprocessor.analyze()
    print(dataset_analysis)


if __name__ == "__main__":
    main()
