"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json
from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_8_llm' / 'settings.json', 'r', encoding='utf-8') as settings:
        settings_dict = json.load(settings)
    raw_dataset = RawDataImporter(settings_dict["parameters"]["dataset"])
    raw_dataset.obtain()
    processed_dataset = RawDataPreprocessor(raw_dataset.raw_data)
    dataset_analysis = processed_dataset.analyze()
    processed_dataset.transform()
    dataset = TaskDataset(processed_dataset.data.head(100))
    pipeline = LLMPipeline(settings_dict['parameters']['model'], dataset, 120, 64, 'cpu')
    model_analysis = pipeline.analyze_model()
    inf_sample = pipeline.infer_sample(dataset[0])
    predictions = pipeline.infer_dataset()

    print(dataset_analysis)
    print(model_analysis)
    print(predictions)

    result = 1
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
