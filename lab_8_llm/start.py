"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from random import randint

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / 'lab_8_llm' / 'settings.json')
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model, dataset, 120, 64, 'cpu')
    pipeline.analyze_model()

    pipeline.infer_sample(dataset[randint(0, len(dataset) - 1)])

    df_pred = pipeline.infer_dataset()
    predictions_path = PROJECT_ROOT / 'lab_8_llm' / 'dist'
    if not predictions_path.exists():
        predictions_path.mkdir(exist_ok=True)

    df_pred.to_csv(predictions_path / 'predictions.csv', index=False, encoding='utf-8')

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print(result)
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
