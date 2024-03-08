"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
import json

from config.constants import PROJECT_ROOT
from core_utils.llm.time_decorator import report_time
from lab_8_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None
    with open(PROJECT_ROOT / 'lab_8_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()
    assert importer.raw_data is not None, "Demo does not work correctly"

    preprocessor = RawDataPreprocessor(importer.raw_data)
    print(preprocessor.analyze())
    preprocessor.transform()
    print(preprocessor)
    print(preprocessor.data)

    dataset = TaskDataset(preprocessor.data.head(10))
    pipline = LLMPipeline(settings['parameters']['model'], dataset, 120, 1, 'cpu')
    print(pipline.analyze_model())
    for i in dataset:
        print(i)
    results = {'0': 'not toxic', '1': 'toxic'}
    texts = ['fuck you, faggot', 'i love you']
    for text in texts:
        result = pipline.infer_sample(text)
        toxic = results[result]
        print(f'Comment:{text}\nToxic: {toxic}')
    #print(f'Comment.{text1}\ntoxic:{result1}')
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
