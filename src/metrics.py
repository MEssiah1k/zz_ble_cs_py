"""Performance metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def absolute_error(true_d: float, est_d: float) -> float:
    return float(abs(true_d - est_d))


def squared_error(true_d: float, est_d: float) -> float:
    diff = true_d - est_d
    return float(diff * diff)


def rmse(true_list: list[float] | np.ndarray, est_list: list[float] | np.ndarray) -> float:
    true_arr = np.asarray(true_list, dtype=float)
    est_arr = np.asarray(est_list, dtype=float)
    return float(np.sqrt(np.mean((true_arr - est_arr) ** 2)))


def mae(true_list: list[float] | np.ndarray, est_list: list[float] | np.ndarray) -> float:
    true_arr = np.asarray(true_list, dtype=float)
    est_arr = np.asarray(est_list, dtype=float)
    return float(np.mean(np.abs(true_arr - est_arr)))


def median_ae(true_list: list[float] | np.ndarray, est_list: list[float] | np.ndarray) -> float:
    true_arr = np.asarray(true_list, dtype=float)
    est_arr = np.asarray(est_list, dtype=float)
    return float(np.median(np.abs(true_arr - est_arr)))


def percentile_ae(true_list: list[float] | np.ndarray, est_list: list[float] | np.ndarray, q: float) -> float:
    true_arr = np.asarray(true_list, dtype=float)
    est_arr = np.asarray(est_list, dtype=float)
    return float(np.percentile(np.abs(true_arr - est_arr), q))


def success_rate(true_list: list[float] | np.ndarray, est_list: list[float] | np.ndarray, threshold: float) -> float:
    true_arr = np.asarray(true_list, dtype=float)
    est_arr = np.asarray(est_list, dtype=float)
    return float(np.mean(np.abs(true_arr - est_arr) <= threshold))


def summarize_errors(true_list: list[float] | np.ndarray, est_list: list[float] | np.ndarray, thresholds: list[float] | None = None) -> dict[str, Any]:
    thresholds = thresholds or [0.1, 0.5, 1.0]
    summary = {
        "mae": mae(true_list, est_list),
        "rmse": rmse(true_list, est_list),
        "median_ae": median_ae(true_list, est_list),
        "p90_ae": percentile_ae(true_list, est_list, 90),
    }
    for threshold in thresholds:
        summary[f"success_rate_le_{threshold}m"] = success_rate(true_list, est_list, threshold)
    return summary
