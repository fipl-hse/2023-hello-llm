"""
Neural machine translation starter.
"""
import json
from pprint import pprint

from config.constants import PROJECT_ROOT
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None
    with open(PROJECT_ROOT / "lab_7_llm" / "settings.json", "r", encoding="utf-8") as settings_json:
        settings = json.load(settings_json)

    raw_data = RawDataImporter(settings["parameters"]["dataset"])
    raw_data.obtain()

    preprocessed_data = RawDataPreprocessor(raw_data.raw_data)
    pprint(preprocessed_data.analyze())

    preprocessed_data.transform()

    dataset = TaskDataset(preprocessed_data.data.head(100))

    llm = LLMPipeline(settings["parameters"]["model"], dataset, 120, 2, "cpu")

    pprint(llm.analyze_model())

    result = llm.infer_sample(dataset[0])
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
