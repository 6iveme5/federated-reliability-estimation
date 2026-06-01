from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def polynomial_calibrate(source: np.ndarray, target: np.ndarray, degree: int = 3) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 1)
    target = np.asarray(target, dtype=np.float64)
    x_poly = PolynomialFeatures(degree=degree).fit_transform(source)
    return LinearRegression().fit(x_poly, target).predict(x_poly)


def isotonic_calibrate(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return IsotonicRegression(out_of_bounds="clip").fit_transform(source, target)
