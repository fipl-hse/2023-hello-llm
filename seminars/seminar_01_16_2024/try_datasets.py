"""
HuggingFace datasets listing.
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def main() -> None:
    """
    Entrypoint for the listing.
    """
    # 1. Obtain dataset from HuggingFace
    dataset = load_dataset(
        'RussianNLP/russian_super_glue',
        name='danetqa'
    )

    # 2. Check dataset's subset
    print(dataset.data.keys())

    # 3. Get needed subset
    subset = dataset.get('validation')

    # 4. Get number of samples
    print(f'Obtained dataset step-by-step: # of samples is {len(subset)}')

    # 5. Get dataset with particular subset at once
    dataset = load_dataset(
        'RussianNLP/russian_super_glue',
        name='danetqa',
        split='validation'
    )
    print(f'Obtained dataset with one call: # of samples is {len(dataset)}')

    # 6. Dataset without a name
    dataset = load_dataset('sberquad', split='validation')
    print(f'Obtained sberquad dataset with one call: # of samples is {len(dataset)}')

    # 7. Cast dataset to pandas
    dataset_df: pd.DataFrame = subset.to_pandas()

    # 8. Optionally save head of dataframe
    (
        dataset_df
        .head(100)
        .to_csv(
            Path(__file__).parent / 'assets' / 'danetqa_example.csv',
            index=False,
            encoding='utf-8'
        )
    )


if __name__ == '__main__':
    main()
