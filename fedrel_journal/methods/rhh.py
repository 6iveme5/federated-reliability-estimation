from __future__ import annotations

import numpy as np

from fedrel_journal.metrics import minmax01


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def random_hyperplanes(dim: int, bits: int = 128, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((bits, dim)).astype(np.float32)


def hash_bits(x: np.ndarray, planes: np.ndarray) -> np.ndarray:
    return (l2_normalize(x) @ planes.T) >= 0


def hamming_distances(query_bits: np.ndarray, train_bits: np.ndarray) -> np.ndarray:
    return np.logical_xor(query_bits[:, None, :], train_bits[None, :, :]).sum(axis=2).astype(
        np.float32
    )


def rhh_scores(
    x_train: np.ndarray,
    x_query: np.ndarray,
    bits: int = 128,
    neighbors: int = 10,
    seed: int = 42,
    alpha: float | None = None,
) -> dict[str, np.ndarray]:
    planes = random_hyperplanes(x_train.shape[1], bits=bits, seed=seed)
    return rhh_scores_with_planes(x_train, x_query, planes=planes, neighbors=neighbors, alpha=alpha)


def rhh_scores_with_planes(
    x_train: np.ndarray,
    x_query: np.ndarray,
    planes: np.ndarray,
    neighbors: int = 10,
    alpha: float | None = None,
) -> dict[str, np.ndarray]:
    train_bits = hash_bits(x_train, planes)
    query_bits = hash_bits(x_query, planes)
    distances = hamming_distances(query_bits, train_bits)
    k = min(neighbors, distances.shape[1])
    topk = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    nearest = topk.min(axis=1)
    if alpha is None:
        median = np.median(topk[:, 0])
        alpha = np.log(2.0) / max(float(median), 1.0)

    exact_counts = (distances == 0).sum(axis=1).astype(np.float64)
    radius = np.percentile(distances, 5)
    adaptive_counts = (distances <= radius).sum(axis=1).astype(np.float64)
    return {
        "s_nn_global": minmax01(-nearest).astype(np.float32),
        "s_knn_global": minmax01(-topk.mean(axis=1)).astype(np.float32),
        "s_kernel_global": minmax01(np.exp(-alpha * topk).mean(axis=1)).astype(np.float32),
        "s_count_global": minmax01(exact_counts).astype(np.float32),
        "s_count_adapt_global": minmax01(adaptive_counts).astype(np.float32),
    }


def fed_rhh_scores(
    clients: list[np.ndarray],
    x_query: np.ndarray,
    bits: int = 128,
    neighbors: int = 10,
    seed: int = 42,
    alpha: float | None = None,
) -> dict[str, np.ndarray]:
    planes = random_hyperplanes(x_query.shape[1], bits=bits, seed=seed)
    local = [
        rhh_scores_with_planes(x_client, x_query, planes=planes, neighbors=neighbors, alpha=alpha)
        for x_client in clients
    ]
    weights = np.asarray([len(x) for x in clients], dtype=np.float64)
    weights = weights / weights.sum()
    combined = {}
    for key in local[0]:
        score = sum(weight * scores[key] for weight, scores in zip(weights, local, strict=True))
        combined[key] = minmax01(score).astype(np.float32)
    return combined


def transfer_rhh_scores(
    clients: list[np.ndarray],
    x_query: np.ndarray,
    bits: int = 128,
    neighbors: int = 10,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    x_train = np.vstack(clients)
    cl_scores = rhh_scores(x_train, x_query, bits=bits, neighbors=neighbors, seed=seed)
    fl_scores = fed_rhh_scores(clients, x_query, bits=bits, neighbors=neighbors, seed=seed)
    return cl_scores, fl_scores
