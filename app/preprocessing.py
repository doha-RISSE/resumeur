import re
import nltk
nltk.download("punkt")

from nltk.tokenize import sent_tokenize

def clean_text(text: str):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text: str):
    text = clean_text(text)
    sentences = sent_tokenize(text)

    return [
        {"text": s, "position": i}
        for i, s in enumerate(sentences)
    ]