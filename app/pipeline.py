from app.preprocessing import split_sentences
from app.textrank import textrank
from app.mmr import mmr

def summarize_document(text: str):
    # 1. split sentences
    sentences = split_sentences(text)

    # 2. chunking simple (améliorable)
    chunks = [sentences[i:i+10] for i in range(0, len(sentences), 10)]

    # 3. TextRank local
    local_summaries = []
    for chunk in chunks:
        local_summaries.extend(textrank(chunk, top_k=3))

    # 4. Fusion + MMR
    final_summary = mmr(local_summaries, top_k=5)

    return final_summary