import hdbscan
from app.embeddings import get_embeddings

def semantic_chunking(sentences):
    embeddings = get_embeddings(sentences)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        metric="euclidean"
    )

    labels = clusterer.fit_predict(embeddings)

    clusters = {}

    for i, label in enumerate(labels):
        if label == -1:
            continue  # outliers ignorés
        clusters.setdefault(label, []).append(sentences[i])

    return list(clusters.values())