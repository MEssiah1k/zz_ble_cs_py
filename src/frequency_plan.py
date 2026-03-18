"""Frequency grid helpers."""

from __future__ import annotations

import numpy as np


def build_frequency_grid(f_start: float, f_step: float, n_freqs: int) -> np.ndarray:
    return f_start + f_step * np.arange(n_freqs, dtype=float)


def validate_frequency_spacing(freqs: np.ndarray, tau_max: float) -> dict[str, float | bool]:
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1 or freqs.size < 2:
        return {"valid": True, "max_step_hz": 0.0, "tau_limit_s": float("inf")}
    steps = np.diff(freqs)
    max_step = float(np.max(np.abs(steps)))
    tau_limit = 1.0 / (2.0 * max_step) if max_step > 0 else float("inf")
    return {"valid": tau_max <= tau_limit, "max_step_hz": max_step, "tau_limit_s": tau_limit}
