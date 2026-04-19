import re
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def clean_text(text: str):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_sentences(text: str):
    text = clean_text(text)
    return sent_tokenize(text)