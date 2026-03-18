"""Channel and fading models."""

from __future__ import annotations

import numpy as np

from .constants import SPEED_OF_LIGHT
from .utils import complex_awgn


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128)
    signal_power = float(np.mean(np.abs(signal) ** 2))
    return signal + complex_awgn(signal.shape, snr_db=snr_db, signal_power=signal_power, rng=rng)


def apply_generic_multipath(
    freqs: np.ndarray,
    path_delays: np.ndarray,
    path_gains: np.ndarray,
    path_phases: np.ndarray,
) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    response = np.zeros(freqs.shape[0], dtype=np.complex128)
    for delay, gain, phase in zip(path_delays, path_gains, path_phases):
        response += gain * np.exp(-1j * 2.0 * np.pi * freqs * delay) * np.exp(1j * phase)
    return response


def apply_single_reflector_multipath(freqs: np.ndarray, base_distance: float, extra_paths: list[dict[str, float]]) -> np.ndarray:
    base_delay = 2.0 * base_distance / SPEED_OF_LIGHT
    path_delays = [base_delay]
    path_gains = [1.0]
    path_phases = [0.0]
    for path in extra_paths:
        path_delays.append(base_delay + float(path["delay_offset"]))
        path_gains.append(float(path["amplitude"]))
        path_phases.append(float(path["phase"]))
    return apply_generic_multipath(freqs, np.asarray(path_delays), np.asarray(path_gains), np.asarray(path_phases))


def apply_rician_fading(signal: np.ndarray, k_factor: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    signal = np.asarray(signal, dtype=np.complex128)
    k = max(float(k_factor), 0.0)
    los = np.sqrt(k / (k + 1.0)) if k > 0 else 0.0
    scatter_scale = np.sqrt(1.0 / (2.0 * (k + 1.0)))
    scatter = scatter_scale * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
    return signal * (los + scatter)


def apply_rayleigh_fading(signal: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    signal = np.asarray(signal, dtype=np.complex128)
    fading = (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)) / np.sqrt(2.0)
    return signal * fading
