from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from fedrel_journal.metrics import minmax01

RELIABILITY_CLIP_QUANTILES = (0.01, 0.99)


@dataclass(frozen=True)
class ReliabilityPostprocessParams:
    fill_value: float
    clip_low: float
    clip_high: float
    logged_min: float
    logged_max: float


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
    scores_array = classaware_hash_raw_scores(
        x_train=x_train,
        y_train=y_train,
        x_query=x_query,
        query_pred=query_pred,
        k=k,
        n_hash=n_hash,
        seed=seed,
    )
    if postprocess:
        return postprocess_reliability(scores_array)
    return minmax01(scores_array).astype(np.float32)


def classaware_hash_raw_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    query_pred: np.ndarray,
    k: int = 10,
    n_hash: int = 128,
    seed: int = 42,
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

    return np.asarray(scores, dtype=np.float32)


def postprocess_reliability(
    scores: np.ndarray,
    clip_quantiles: tuple[float, float] = RELIABILITY_CLIP_QUANTILES,
) -> np.ndarray:
    params = fit_reliability_postprocess(scores, clip_quantiles=clip_quantiles)
    return apply_reliability_postprocess(scores, params)


def fit_reliability_postprocess(
    scores: np.ndarray,
    clip_quantiles: tuple[float, float] = RELIABILITY_CLIP_QUANTILES,
) -> ReliabilityPostprocessParams:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return ReliabilityPostprocessParams(
            fill_value=0.5,
            clip_low=0.5,
            clip_high=0.5,
            logged_min=0.0,
            logged_max=0.0,
        )

    finite_mask = np.isfinite(scores)
    fill_value = float(np.nanmedian(scores[finite_mask])) if finite_mask.any() else 0.5
    scores = scores.copy()
    scores[~finite_mask] = fill_value

    if scores.size > 1 and float(np.max(scores)) > float(np.min(scores)):
        low, high = np.quantile(scores, clip_quantiles)
        clip_low = float(low)
        clip_high = float(high)
    else:
        clip_low = float(scores[0]) if scores.size else fill_value
        clip_high = clip_low

    clipped = np.clip(scores, clip_low, clip_high)
    shifted = clipped - float(np.min(clipped))
    logged = np.log1p(shifted)
    return ReliabilityPostprocessParams(
        fill_value=fill_value,
        clip_low=clip_low,
        clip_high=clip_high,
        logged_min=float(np.min(logged)) if logged.size else 0.0,
        logged_max=float(np.max(logged)) if logged.size else 0.0,
    )


def apply_reliability_postprocess(
    scores: np.ndarray,
    params: ReliabilityPostprocessParams,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores

    scores = scores.copy()
    scores[~np.isfinite(scores)] = params.fill_value
    scores = np.clip(scores, params.clip_low, params.clip_high)
    shifted = scores - params.clip_low
    logged = np.log1p(shifted)
    denom = params.logged_max - params.logged_min
    if denom <= 1e-12:
        return np.full(scores.shape, 0.5, dtype=np.float32)
    scaled = (logged - params.logged_min) / denom
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _mean_knn_distance(reference: np.ndarray, query: np.ndarray, k: int) -> float:
    n_neighbors = min(k, len(reference))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(reference)
    distances = nn.kneighbors(query.reshape(1, -1), return_distance=True)[0]
    return float(distances.mean())
