from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def mmr(documents, top_k=5, lambda_param=0.7):
    if len(documents) <= top_k:
        return documents

    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    sim_matrix = cosine_similarity(vectors)

    selected = []
    unselected = list(range(len(documents)))

    selected.append(0)
    unselected.remove(0)

    while len(selected) < top_k:
        mmr_scores = []

        for i in unselected:
            relevance = max(sim_matrix[i][j] for j in selected)
            diversity = sum(sim_matrix[i][j] for j in selected) / len(selected)

            score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((score, i))

        best = max(mmr_scores)[1]
        selected.append(best)
        unselected.remove(best)

    return [documents[i] for i in selected]