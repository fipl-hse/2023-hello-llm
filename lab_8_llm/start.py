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

    if not isinstance(importer.raw_data, pd.DataFrame) or importer.raw_data is None:
        raise TypeError

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model, dataset, 120, 64, 'cpu')
    pipeline.analyze_model()

    pipeline.infer_sample(dataset[randint(0, len(dataset) - 1)])

    df_pred = pipeline.infer_dataset()
    predictions_path = PROJECT_ROOT / 'lab_8_llm' / 'dist' / 'predictions.csv'
    if not predictions_path.parent.exists():
        predictions_path.parent.mkdir(exist_ok=True)

    df_pred.to_csv(predictions_path, index=False, encoding='utf-8')

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print(result)
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
