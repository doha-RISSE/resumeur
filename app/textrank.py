import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def textrank(sentences, top_k=5):
    if len(sentences) <= top_k:
        return sentences

    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    sim_matrix = cosine_similarity(vectors)

    graph = nx.from_numpy_array(sim_matrix)

    scores = nx.pagerank(graph)

    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    return [s for _, s in ranked[:top_k]]