"""
Neural machine translation starter.
"""
import json
import os
from pathlib import Path

from config.constants import PROJECT_ROOT
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
# pylint: disable= too-many-locals
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_7_llm' / 'settings.json', "r", encoding='utf-8') as settings_path:
        settings = json.load(settings_path)
    ds_obtained = RawDataImporter(settings["parameters"]["dataset"])
    ds_obtained.obtain()
    preprocessed_ds = RawDataPreprocessor(ds_obtained.raw_data)
    preprocessed_ds.analyze()
    preprocessed_ds.transform()
    task_ds = TaskDataset(preprocessed_ds.data.head(100))
    llm_infer = LLMPipeline(model_name=settings["parameters"]["model"], dataset=task_ds,
                            max_length=120, batch_size=64, device='cpu')
    llm_infer.analyze_model()
    llm_infer.infer_sample(task_ds[0])
    if not os.path.exists(f'{PROJECT_ROOT}/lab_7_llm/dist'):
        os.mkdir(f'{PROJECT_ROOT}/lab_7_llm/dist')
    llm_infer.infer_dataset().to_csv(f'{PROJECT_ROOT}/lab_7_llm/dist/predictions.csv', index=False)
    result = TaskEvaluator(data_path=Path(f'{PROJECT_ROOT}/lab_7_llm/dist/predictions.csv'),
                           metrics=Metrics)

    result.run()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
