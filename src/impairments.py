"""Lightweight phase impairment helpers."""

from __future__ import annotations

import numpy as np


def fixed_phase_offset(value: float, n_freqs: int) -> np.ndarray:
    return np.full(n_freqs, float(value))


def random_phase_offsets(
    n_repeats: int,
    n_devices: int,
    n_freqs: int,
    rng: np.random.Generator,
    low: float = 0.0,
    high: float = 2.0 * np.pi,
) -> np.ndarray:
    return rng.uniform(low, high, size=(n_repeats, n_devices, n_freqs))


def frequency_dependent_phase(freqs: np.ndarray, slope_rad_per_hz: float = 0.0, intercept: float = 0.0) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    return intercept + slope_rad_per_hz * (freqs - freqs[0])
