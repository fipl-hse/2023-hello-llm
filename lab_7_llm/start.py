"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json

from config.constants import PROJECT_ROOT
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT/'lab_7_llm'/'settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(settings['parameters']['model'],
                           dataset,
                           120,
                           2,
                           'cpu')
    pipeline.analyze_model()
    pipeline.infer_sample(dataset[0])
    result = pipeline.infer_dataset()

    res_path = PROJECT_ROOT/'lab_7_llm'/'dist'
    if not res_path.exists():
        res_path.mkdir()
    result.to_csv(res_path/'predictions.csv', index=False)
    result = TaskEvaluator(data_path=res_path/'predictions.csv',
                           metrics=[Metrics[metric.upper()] for metric in
                                    settings['parameters']['metrics']])

    print(result.run())

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
