"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals

from pathlib import Path

import json
import os
from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (RawDataImporter, RawDataPreprocessor,
                            LLMPipeline, TaskDataset, TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)
    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.get_raw_data)
    dataset_analysis = preprocessor.analyze()
    print(dataset_analysis)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(model_name=settings['parameters']['model'], dataset=dataset,
                           max_length=120,
                           batch_size=64,
                           device='cpu')

    analysis = pipeline.analyze_model()
    print(analysis)
    infer_sample_result = pipeline.infer_sample(dataset[0])
    print(infer_sample_result)

    if not os.path.exists(PROJECT_ROOT / 'lab_7_llm' / 'dist'):
        os.mkdir(PROJECT_ROOT / 'lab_7_llm' / 'dist')

    path_to_predictions = PROJECT_ROOT / 'lab_7_llm' / 'dist' / 'predictions.csv'
    pipeline.infer_dataset().to_csv(path_to_predictions, index=False)

    evaluation = TaskEvaluator(data_path=Path(path_to_predictions),
                               metrics=settings['parameters']['metrics'])
    result = evaluation.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
