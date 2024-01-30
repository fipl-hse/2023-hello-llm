"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset
from config.constants import PROJECT_ROOT
import json

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', "r")
    settings = json.load(settings_path)
    ds = RawDataImporter(settings["parameters"]["dataset"])
    ds.obtain()
    preprocessed_ds = RawDataPreprocessor(ds.raw_data)
    preprocessed_ds.analyze()
    preprocessed_ds.transform()
    result = TaskDataset(preprocessed_ds.data)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
