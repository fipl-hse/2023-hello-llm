"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json
from pathlib import Path

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_8_llm' / 'settings.json', 'r', encoding='utf-8') as settings_file:
        settings = json.load(settings_file)

    pred_path = PROJECT_ROOT / "dist" / "predictions.csv"

    if not pred_path.parent.exists():
        pred_path.parent.mkdir(exist_ok=True)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    batch_size = 1
    max_length = 120

    pipeline = LLMPipeline(model_name=settings['parameters']['model'],
                           dataset=dataset,
                           max_length=max_length,
                           batch_size=batch_size,
                           device="cpu")
    pipeline.analyze_model()
    pipeline.infer_sample(next(iter(dataset)))

    batch_size = 64
    pipeline = LLMPipeline(model_name=settings['parameters']['model'],
                           dataset=dataset,
                           max_length=max_length,
                           batch_size=batch_size,
                           device="cpu")
    predictions = pipeline.infer_dataset()
    predictions.to_csv(pred_path, index=False, encoding="utf-8")

    evaluator = TaskEvaluator(pred_path, settings['parameters']['metrics'])
    result = evaluator.run()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
