# Project 5 — Sentiment Analyzer v2
# Uses a pre-trained BERT model via HuggingFace Transformers
# to classify text sentiment with high accuracy.

from transformers import pipeline


def load_model():
    classifier = pipeline("sentiment-analysis")
    return classifier


def analyze_texts(classifier, texts):
    results = classifier(texts)
    return results


def display_results(texts, results):
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} ({result['score'] * 100:.2f}%)")


if __name__ == "__main__":
    classifier = load_model()
    texts = [
        "It was not as bad as I expected.",
        "It was not so horrible.", 
        "I could come back anytime.",
        "Best experience of all time.",
        "Horrible destination.",
        "I hope they come back soon."
    ]

    results = analyze_texts(classifier, texts)
    display_results(texts, results)