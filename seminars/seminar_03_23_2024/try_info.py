"""
HuggingFace model listing.
"""

try:
    from transformers import AutoModelForSequenceClassification
except ImportError:
    print('Library "pandas" not installed. Failed to import.')

try:
    from torchinfo import summary
except ImportError:
    print('Library "torchinfo" not installed. Failed to import.')

try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for the listing.
    """

    # 1. Import model
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/russian_toxicity_classifier")

    # 2. Get model config
    config = model.config

    # 3.
    print(config)

    # 4.
    embeddings_length = config.max_position_embeddings

    # 5.
    ids = torch.ones(1, embeddings_length, dtype=torch.long)

    # 6.
    tokens = {
        'input_ids': ids,
        'attention_mask': ids
    }
    # 7.
    result = summary(model,
                     input_data=tokens,
                     device='cpu',
                     verbose=0)

    # 8.
    print(result)


if __name__ == '__main__':
    main()
