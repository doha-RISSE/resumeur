import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.embeddings import model

def mmr(sentences, top_k=5, lambda_param=0.7):

    if len(sentences) <= top_k:
        return sentences

    texts = [s["text"] for s in sentences]
    embeddings = model.encode(texts)

    sim = cosine_similarity(embeddings)

    selected = []
    unselected = list(range(len(sentences)))

    selected.append(0)
    unselected.remove(0)

    while len(selected) < top_k:

        scores = []

        for i in unselected:
            relevance = max(sim[i][j] for j in selected)
            diversity = np.mean([sim[i][j] for j in selected])

            score = lambda_param * relevance - (1 - lambda_param) * diversity
            scores.append((score, i))

        best = max(scores)[1]
        selected.append(best)
        unselected.remove(best)

    return [sentences[i] for i in selected]