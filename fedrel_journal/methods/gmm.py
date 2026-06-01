from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from fedrel_journal.metrics import minmax01


def fit_cl_gmm(x_train: np.ndarray, n_components: int = 8, seed: int = 42) -> GaussianMixture:
    x_train = np.asarray(x_train, dtype=np.float64)
    max_components = max(1, min(n_components, len(x_train) // 2))
    last_error: Exception | None = None
    for k in range(max_components, 0, -1):
        try:
            return GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=1e-4,
                random_state=seed,
                max_iter=300,
            ).fit(x_train)
        except ValueError as exc:
            last_error = exc
    raise RuntimeError("GMM fitting failed even with one component") from last_error


def score_cl_gmm(model: GaussianMixture, x_query: np.ndarray) -> np.ndarray:
    return minmax01(model.score_samples(np.asarray(x_query, dtype=np.float64))).astype(np.float32)


def fit_fed_gmm_mixture(
    clients: list[np.ndarray],
    n_components: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = sum(len(x_client) for x_client in clients)
    if total <= 0:
        raise ValueError("clients must contain at least one sample")

    weights = []
    means = []
    variances = []
    for idx, x_client in enumerate(clients):
        local_components = min(n_components, max(1, len(x_client)))
        model = fit_cl_gmm(x_client, local_components, seed + idx)
        weights.append(model.weights_ * (len(x_client) / total))
        means.append(model.means_)
        variances.append(model.covariances_)
    return np.concatenate(weights), np.vstack(means), np.vstack(variances)


def score_diag_gmm(
    x_query: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    x_query = np.asarray(x_query, dtype=np.float64)
    x = x_query[:, None, :]
    mu = means[None, :, :]
    var = np.maximum(variances[None, :, :], 1e-8)
    d = x_query.shape[1]
    log_prob = -0.5 * (
        d * np.log(2.0 * np.pi)
        + np.log(var).sum(axis=2)
        + (((x - mu) ** 2) / var).sum(axis=2)
    )
    log_weighted = np.log(np.maximum(weights, 1e-300))[None, :] + log_prob
    return minmax01(logsumexp(log_weighted, axis=1)).astype(np.float32)


def transfer_gmm_scores(
    clients: list[np.ndarray],
    x_query: np.ndarray,
    n_components: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.vstack(clients)
    cl_model = fit_cl_gmm(x_train, n_components=n_components, seed=seed)
    cl_scores = score_cl_gmm(cl_model, x_query)
    weights, means, variances = fit_fed_gmm_mixture(clients, n_components=n_components, seed=seed)
    fl_scores = score_diag_gmm(x_query, weights=weights, means=means, variances=variances)
    return cl_scores, fl_scores
