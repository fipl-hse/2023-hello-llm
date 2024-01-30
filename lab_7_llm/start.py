"""
Neural machine translation starter.
"""
import json

from config.constants import PROJECT_ROOT
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline


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
    task_ds = TaskDataset(preprocessed_ds.data.head(100))
    result = LLMPipeline(model_name=settings["parameters"]["model"], dataset=task_ds,
                         max_length=512, batch_size=1, device='cpu')
    result.analyze_model()
    print(result.infer_sample(task_ds.__getitem__(0)))
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
