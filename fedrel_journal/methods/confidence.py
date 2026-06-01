from __future__ import annotations

import numpy as np

from fedrel_journal.metrics import minmax01


def pmax_reliability(proba: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(proba, dtype=np.float64), axis=1).astype(np.float32)


def margin_reliability(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    if proba.shape[1] == 1:
        return np.ones(proba.shape[0], dtype=np.float32)
    sorted_proba = np.sort(proba, axis=1)
    return (sorted_proba[:, -1] - sorted_proba[:, -2]).astype(np.float32)


def entropy_reliability(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    entropy = -(proba * np.log(np.maximum(proba, 1e-12))).sum(axis=1)
    return minmax01(-entropy).astype(np.float32)
