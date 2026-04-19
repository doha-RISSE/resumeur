import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from app.embeddings import model
def textrank(sentences, top_k=5):
    if len(sentences) <= top_k:
        return sentences

    texts = [s["text"] for s in sentences]

    embeddings = model.encode(texts)

    sim_matrix = cosine_similarity(embeddings)

    graph = nx.from_numpy_array(sim_matrix)

    scores = nx.pagerank(graph)

    ranked = sorted(
        ((scores[i], sentences[i]) for i in range(len(sentences))),
        reverse=True
    )

    return [s for _, s in ranked[:top_k]]