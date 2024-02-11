"""
Neural machine translation starter.
"""

import json

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset, TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as settings:
        settings = json.load(settings)
    govreport = RawDataImporter(hf_name=settings["parameters"]["dataset"])
    govreport.obtain()
    preprocessor = RawDataPreprocessor(govreport.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings['parameters']['model'],
                           dataset,
                           max_length=120,
                           batch_size=1,
                           device='cpu')

    pipeline.analyze_model()
    result = pipeline.infer_sample(dataset[0])
    print(result)

    """pred_path = PROJECT_ROOT / "lab_7_llm" / "dist"

    if not pred_path.exists():
        pred_path.mkdir()

    pipeline2 = LLMPipeline(model_name=settings["parameters"]["model"],
                                   dataset=dataset,
                                   max_length=120,
                                   batch_size=64,
                                   device="cpu")

    pipeline2.infer_dataset().to_csv(pred_path, "predictions.csv", index=False)

    evaluator = TaskEvaluator(data_path=pred_path, metrics=settings["parameters"]["metrics"])

    result = evaluator.run()"""

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
