from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def minmax01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    if not np.isfinite(values).any():
        return np.full_like(values, 0.5, dtype=np.float64)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if high <= low:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - low) / (high - low)


def safe_corr(kind: str, reference: np.ndarray, predicted: np.ndarray) -> float:
    reference = np.asarray(reference).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    if np.std(reference) < 1e-12 or np.std(predicted) < 1e-12:
        return float("nan")
    fn = spearmanr if kind == "spearman" else pearsonr
    return float(fn(reference, predicted)[0])


def approximation_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    mask = np.isfinite(reference) & np.isfinite(predicted)
    reference = reference[mask]
    predicted = predicted[mask]
    if len(reference) == 0:
        return {
            "pearson": float("nan"),
            "spearman": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
        }
    return {
        "pearson": safe_corr("pearson", reference, predicted),
        "spearman": safe_corr("spearman", reference, predicted),
        "mae": float(mean_absolute_error(reference, predicted)),
        "rmse": float(mean_squared_error(reference, predicted) ** 0.5),
        "r2": float(r2_score(reference, predicted)),
    }


def error_detection_metrics(
    error_labels: np.ndarray,
    reliability_scores: np.ndarray,
) -> dict[str, float]:
    error_labels = np.asarray(error_labels).reshape(-1)
    error_scores = -np.asarray(reliability_scores)
    if len(np.unique(error_labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(error_labels, error_scores)),
        "auprc": float(average_precision_score(error_labels, error_scores)),
    }


def risk_at_fraction(
    errors: np.ndarray,
    reliability_scores: np.ndarray,
    fraction: float = 0.05,
) -> float:
    errors = np.asarray(errors)
    reliability_scores = np.asarray(reliability_scores)
    n_selected = max(1, int(np.ceil(len(reliability_scores) * fraction)))
    selected = np.argsort(reliability_scores)[:n_selected]
    return float(errors[selected].mean())
