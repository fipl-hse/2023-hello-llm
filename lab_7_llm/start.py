"""
Neural machine translation starter.
"""
import json
import os
from pathlib import Path

from config.constants import PROJECT_ROOT
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor,
                            TaskDataset, TaskEvaluator)


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
    llm_infer = LLMPipeline(model_name=settings["parameters"]["model"], dataset=task_ds,
                            max_length=120, batch_size=64, device='cpu')
    llm_infer.analyze_model()
    llm_infer.infer_sample(task_ds.__getitem__(0))
    if not os.path.exists('./dist'):
        os.mkdir('./dist')
    llm_infer.infer_dataset().to_csv('./dist/predictions.csv', index=False)
    result = TaskEvaluator(data_path=Path('./dist/predictions.csv'),
                           metrics=settings['parameters']['metrics'])
    result.run()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
