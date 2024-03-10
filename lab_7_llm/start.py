"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
import json

from config.constants import PROJECT_ROOT
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    path_settings = PROJECT_ROOT / "lab_7_llm" / "settings.json"
    device = "cpu"
    batch_size = 1
    max_length = 120
    num_samples = 10

    with open(path_settings, encoding="utf-8") as path:
        settings = json.load(path)
    data_loader = RawDataImporter(settings["parameters"]["dataset"])
    data_loader.obtain()

    preprocessor = RawDataPreprocessor(data_loader.raw_data)
    dataset_analysis = preprocessor.analyze()
    print(dataset_analysis)

    # 6
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(num_samples))
    pipeline = LLMPipeline(settings["parameters"]["model"], dataset, max_length, batch_size, device)

    analysis = pipeline.analyze_model()
    print(analysis)

    result = pipeline.infer_sample(dataset[0])
    print('EN:', dataset[0][0])
    print('FR:', result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
