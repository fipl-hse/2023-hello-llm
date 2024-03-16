"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json
from pprint import pprint

from core_utils.llm.time_decorator import report_time

from config.constants import PROJECT_ROOT
from lab_7_llm.main import (
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    LLMPipeline,
)

SETTINGS = PROJECT_ROOT / "lab_7_llm" / "settings.json"


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(SETTINGS, encoding="utf-8") as path:
        settings = json.load(path)
    data_loader = RawDataImporter(settings["parameters"]["dataset"])
    data_loader.obtain()

    preprocessor = RawDataPreprocessor(data_loader.raw_data)
    pprint(preprocessor.analyze())
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(
        settings["parameters"]["model"],
        dataset,
        max_length=120,
        batch_size=64,
        device="cpu",
    )

    result = pipeline.analyze_model()
    print(result)
    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()
