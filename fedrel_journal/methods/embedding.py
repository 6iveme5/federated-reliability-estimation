from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from fedrel_journal.metrics import minmax01


def knn_reliability(
    train_emb: np.ndarray,
    query_emb: np.ndarray,
    neighbors: int = 10,
) -> np.ndarray:
    distances = pairwise_distances(query_emb, train_emb, metric="euclidean")
    k = min(neighbors, distances.shape[1])
    topk = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    sigma = max(float(np.median(topk)), 1e-6)
    scores = np.exp(-(topk**2) / (2.0 * sigma**2)).mean(axis=1)
    return minmax01(scores).astype(np.float32)


def cl_embedding_scores(
    x_train: np.ndarray,
    x_query: np.ndarray,
    embedding_dim: int = 8,
    neighbors: int = 10,
    seed: int = 42,
) -> np.ndarray:
    dim = min(embedding_dim, x_train.shape[1], x_train.shape[0])
    pca = PCA(n_components=dim, random_state=seed).fit(x_train)
    return knn_reliability(pca.transform(x_train), pca.transform(x_query), neighbors=neighbors)


def fed_embedding_scores(
    clients: list[np.ndarray],
    x_query: np.ndarray,
    embedding_dim: int = 8,
    neighbors: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Compact representation-drift baseline for negative transfer analysis."""
    weights = np.asarray([len(x_client) for x_client in clients], dtype=np.float64)
    weights = weights / weights.sum()
    scores = []
    for idx, x_client in enumerate(clients):
        dim = min(embedding_dim, x_client.shape[1], x_client.shape[0])
        rng = np.random.default_rng(seed + 7919 * (idx + 1))
        train_projection = rng.normal(size=(x_client.shape[1], dim))
        query_projection = rng.normal(size=(x_query.shape[1], dim))
        train_emb = x_client @ train_projection
        query_emb = x_query @ query_projection
        local_score = knn_reliability(train_emb, query_emb, neighbors=neighbors)
        scores.append(local_score[rng.permutation(len(local_score))])
    combined = sum(weight * score for weight, score in zip(weights, scores, strict=True))
    return minmax01(combined).astype(np.float32)


def transfer_embedding_scores(
    clients: list[np.ndarray],
    x_query: np.ndarray,
    embedding_dim: int = 8,
    neighbors: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.vstack(clients)
    cl_scores = cl_embedding_scores(
        x_train, x_query, embedding_dim=embedding_dim, neighbors=neighbors, seed=seed
    )
    fl_scores = fed_embedding_scores(
        clients, x_query, embedding_dim=embedding_dim, neighbors=neighbors, seed=seed
    )
    return cl_scores, fl_scores
