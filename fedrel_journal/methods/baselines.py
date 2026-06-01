from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from fedrel_journal.metrics import minmax01
from fedrel_journal.methods.confidence import entropy_reliability, margin_reliability, pmax_reliability


def confidence_baselines(proba: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "pmax": pmax_reliability(proba),
        "entropy": entropy_reliability(proba),
        "margin": margin_reliability(proba),
    }


def centroid_reliability(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    query_pred: np.ndarray,
) -> np.ndarray:
    centroids = {
        label: x_train[y_train == label].mean(axis=0)
        for label in np.unique(y_train)
        if np.any(y_train == label)
    }
    distances = []
    fallback = float(np.linalg.norm(x_train - x_train.mean(axis=0), axis=1).mean())
    for x_i, pred_i in zip(x_query, query_pred, strict=True):
        centroid = centroids.get(pred_i)
        if centroid is None:
            distances.append(fallback)
        else:
            distances.append(float(np.linalg.norm(x_i - centroid)))
    return minmax01(-np.asarray(distances, dtype=np.float64)).astype(np.float32)


def feature_knn_classaware_reliability(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    query_pred: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    scores = []
    for x_i, pred_i in zip(x_query, query_pred, strict=True):
        same = x_train[y_train == pred_i]
        other = x_train[y_train != pred_i]
        if len(same) == 0 or len(other) == 0:
            scores.append(0.5)
            continue
        d_same = mean_knn_distance(same, x_i, k)
        d_other = mean_knn_distance(other, x_i, k)
        scores.append(d_other / (d_same + d_other + 1e-8))
    return minmax01(np.asarray(scores, dtype=np.float64)).astype(np.float32)


def lof_reliability(
    x_train: np.ndarray,
    x_query: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    n_neighbors = min(max(2, k), len(x_train) - 1)
    if n_neighbors < 2:
        return np.full(len(x_query), 0.5, dtype=np.float32)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(x_train)
    return minmax01(lof.score_samples(x_query)).astype(np.float32)


def mean_knn_distance(reference: np.ndarray, query: np.ndarray, k: int) -> float:
    n_neighbors = min(k, len(reference))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(reference)
    distances = nn.kneighbors(query.reshape(1, -1), return_distance=True)[0]
    return float(distances.mean())
