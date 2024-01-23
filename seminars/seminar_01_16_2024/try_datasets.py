"""
HuggingFace datasets listing.
"""
from pathlib import Path

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = None  # type: ignore

try:
    from datasets import load_dataset
except ImportError:
    print('Library "datasets" not installed. Failed to import.')
    load_dataset = None  # type: ignore


def main() -> None:
    """
    Entrypoint for the listing.
    """
    # 1. Obtain dataset from HuggingFace
    dataset = load_dataset(
        'ag_news',
        split='test'
    )
    print(f'Obtained dataset with one call: # of samples is {len(dataset)}')

    # 4. Get number of samples
    print(f'Obtained dataset step-by-step: # of samples is {len(dataset)}')

    # 7. Cast dataset to pandas
    dataset_df: DataFrame = dataset.to_pandas()

    print(dataset_df.columns)

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
