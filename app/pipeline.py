from app.preprocessing import split_sentences
from app.chunking import semantic_chunking
from app.mmr import mmr
from app.textrank import textrank
from app.embeddings import model

import re

def remove_noise_sentences(sentences):
    cleaned = []

    for s in sentences:
        text = s["text"].strip()

        # supprime "3." "4." "5." etc
        if re.fullmatch(r"\d+\.?", text):
            continue

        # supprime aussi les phrases vides
        if len(text) < 3:
            continue

        cleaned.append(s)

    return cleaned

def summarize_document(text: str):

    # 1. preprocessing
    sentences = split_sentences(text)

    # 2. semantic chunking (HDBSCAN)
    chunks = semantic_chunking(sentences)

    # 3. local ranking (TextRank SBERT)
    candidates = []

    for chunk in chunks:
        candidates.extend(textrank(chunk, top_k=3))

    # 4. global selection (MMR)
    final = mmr(candidates, top_k=5)

    # 5. reorder by original position
    final = sorted(final, key=lambda x: x["position"])
    final = remove_noise_sentences(final)
    return [s["text"] for s in final]

