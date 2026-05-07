from __future__ import annotations

import numpy as np


def brier_score(probabilities: list[float], outcomes: list[bool]) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(probabilities: list[float], outcomes: list[bool], n_bins: int = 10) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask) * abs(np.mean(p[mask]) - np.mean(y[mask])))
    return ece


def log_score(probabilities: list[float], outcomes: list[bool], epsilon: float = 1e-12) -> float:
    """Mean negative log likelihood with probability clipping. Lower is better."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return float("nan")
    p = np.clip(p, epsilon, 1.0 - epsilon)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
