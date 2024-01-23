"""
HuggingFace model listing.
"""

try:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    print('Library "pandas" not installed. Failed to import.')

try:
    import numpy as np
except ImportError:
    print('Library "numpy" not installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for the listing.
    """

    # 1. Classification
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/russian_toxicity_classifier")

    # 2. Convert text to tokens
    text = "Простое предложение"
    tokens = tokenizer(text, return_tensors='pt')

    # 3. Print tokens keys
    print(tokens.keys())

    # 4. Import model
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/russian_toxicity_classifier")

    # 5 Print model
    print(model)

    # 6. Сlassify text
    output = model(**tokens)

    # 7. Print prediction
    print(output)

    # 8. Print lable
    predictions = np.argmax(output.logits.detach().numpy(), axis=-1)

    # 9. Print predictions
    print(predictions)

    # 10. Map with labels
    labels = model.config.id2label
    print(labels[predictions[0]])

    # 11. Import tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # 12. Convert text to tokens
    text = "Generate any text"
    tokens = tokenizer(text, return_tensors='pt')

    # 13. Print tokens keys
    print(tokens.keys())

    # 14. Load model
    model = AutoModelForCausalLM.from_pretrained('gpt2')

    # 15. Print model
    print(model)

    # 16. Generate text
    output = model.generate(**tokens)
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(results[0])


if __name__ == '__main__':
    main()
