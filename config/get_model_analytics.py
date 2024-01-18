"""
Collects and stores model analytics
"""
import json
from pathlib import Path
from typing import Any

import pandas as pd

from reference_lab_nmt.main import LLMPipeline, TaskDataset


def get_references(path: Path) -> Any:
    """
    Load reference_scores.json file
    """
    with open(path, encoding='utf-8') as file:
        return json.load(file)


def save_reference(path: Path, refs: dict) -> None:
    """
    Save analytics
    """
    with open(path, mode='w', encoding='utf-8') as file:
        json.dump(
            refs,
            file,
            indent=4,
            ensure_ascii=True,
            sort_keys=True
        )
    with open(path, mode='a', encoding='utf-8') as file:
        file.write('\n')


def main() -> None:
    """
    Run the collect models analytics.
    """
    batch_size = 64
    max_length = 120
    device = 'cpu'

    references_path = Path(__file__).parent / 'reference_scores.json'
    dest = Path(__file__).parent / 'reference_model_analytics.json'

    references = get_references(path=references_path)
    result = {}
    for model, _ in references.items():
        pipeline = LLMPipeline(model, TaskDataset(pd.DataFrame([])), max_length, batch_size, device)
        print(model)
        model_analysis = pipeline.analyze_model()
        result[model] = model_analysis
    save_reference(dest, result)


if __name__ == '__main__':
    main()
