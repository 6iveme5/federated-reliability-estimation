from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from fedrel_journal.metrics import minmax01

RELIABILITY_CLIP_QUANTILES = (0.01, 0.99)


def pstable_projection(x: np.ndarray, n_hash: int = 128, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(x.shape[1], n_hash)).astype(np.float32)
    return x.astype(np.float32) @ projections


def classaware_hash_reliability(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    query_pred: np.ndarray,
    k: int = 10,
    n_hash: int = 128,
    seed: int = 42,
    postprocess: bool = True,
) -> np.ndarray:
    train_hash = pstable_projection(x_train, n_hash=n_hash, seed=seed)
    query_hash = pstable_projection(x_query, n_hash=n_hash, seed=seed)

    scores = []
    for x_i, pred_i in zip(query_hash, query_pred, strict=True):
        same = train_hash[y_train == pred_i]
        other = train_hash[y_train != pred_i]
        if len(same) == 0 or len(other) == 0:
            scores.append(0.5)
            continue

        d_same = _mean_knn_distance(same, x_i, k)
        d_other = _mean_knn_distance(other, x_i, k)
        scores.append(d_other / (d_same + d_other + 1e-8))

    scores_array = np.asarray(scores, dtype=np.float32)
    if postprocess:
        return postprocess_reliability(scores_array)
    return minmax01(scores_array).astype(np.float32)


def postprocess_reliability(
    scores: np.ndarray,
    clip_quantiles: tuple[float, float] = RELIABILITY_CLIP_QUANTILES,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores

    finite_mask = np.isfinite(scores)
    if not finite_mask.all():
        fill_value = float(np.nanmedian(scores[finite_mask])) if finite_mask.any() else 0.5
        scores = scores.copy()
        scores[~finite_mask] = fill_value

    if scores.size > 1 and float(np.max(scores)) > float(np.min(scores)):
        low, high = np.quantile(scores, clip_quantiles)
        scores = np.clip(scores, low, high)

    shifted = scores - float(np.min(scores))
    logged = np.log1p(shifted)
    return minmax01(logged).astype(np.float32)


def _mean_knn_distance(reference: np.ndarray, query: np.ndarray, k: int) -> float:
    n_neighbors = min(k, len(reference))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(reference)
    distances = nn.kneighbors(query.reshape(1, -1), return_distance=True)[0]
    return float(distances.mean())
