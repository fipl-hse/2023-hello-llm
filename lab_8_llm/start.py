"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import os

import pandas

from config.constants import PROJECT_ROOT
from config.lab_settings import LabSettings
from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset,
                            TaskEvaluator)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(PROJECT_ROOT / "lab_8_llm" / 'settings.json')

    ds_obtainer = RawDataImporter(settings.parameters.dataset)
    ds_obtainer.obtain()

    ds_preprocess = RawDataPreprocessor(ds_obtainer.raw_data)
    print(ds_preprocess.analyze())
    ds_preprocess.transform()

    ds_torch = TaskDataset(ds_preprocess.data.head(100))

    inference_pipeline = LLMPipeline(model_name=settings.parameters.model, dataset=ds_torch,
                                     max_length=120, batch_size=64, device='cpu')
    print(inference_pipeline.analyze_model())
    print(inference_pipeline.infer_sample(ds_torch[0]))

    pred_ds = inference_pipeline.infer_dataset()

    ds_save = PROJECT_ROOT / 'lab_8_llm' / 'dist'

    if not os.path.exists(ds_save):
        os.mkdir(ds_save)
    pred_ds.to_csv(ds_save / 'predictions.csv', index=False, encoding='UTF-8')

    eval = TaskEvaluator(ds_save / 'predictions.csv', Metrics)
    result = eval.run()
    print(result)
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
