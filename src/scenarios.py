"""Scenario generation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import db_to_linear, maybe_repeat


def build_single_device_scenario(distance: float, amplitude: float = 1.0, phase_offset: float = 0.0) -> dict[str, Any]:
    return {
        "distances": np.array([distance], dtype=float),
        "amplitudes": np.array([amplitude], dtype=float),
        "phase_offsets": np.array([phase_offset], dtype=float),
    }


def build_multi_device_scenario(
    num_devices: int,
    distance_mode: str = "uniform",
    distance_range: tuple[float, float] = (1.0, 10.0),
    amplitudes: float | list[float] = 1.0,
    random_phase: bool = True,
    rng: np.random.Generator | None = None,
    specified_distances: list[float] | None = None,
) -> dict[str, Any]:
    rng = rng or np.random.default_rng()
    if specified_distances is not None:
        distances = np.asarray(specified_distances, dtype=float)
        if distances.shape[0] != num_devices:
            raise ValueError("specified_distances length must match num_devices")
    elif distance_mode == "uniform":
        distances = np.linspace(distance_range[0], distance_range[1], num_devices)
    elif distance_mode == "random":
        distances = np.sort(rng.uniform(distance_range[0], distance_range[1], size=num_devices))
    else:
        raise ValueError(f"Unsupported distance_mode: {distance_mode}")
    phases = rng.uniform(0.0, 2.0 * np.pi, size=num_devices) if random_phase else np.zeros(num_devices)
    return {
        "distances": distances,
        "amplitudes": maybe_repeat(amplitudes, num_devices),
        "phase_offsets": phases,
    }


def build_multipath_paths(
    n_paths: int,
    base_delay_s: float,
    delay_spread_s: float,
    gain_db_values: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict[str, float]]:
    rng = rng or np.random.default_rng()
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    gains_db = gain_db_values or [0.0] + [-6.0] * max(n_paths - 1, 0)
    if len(gains_db) < n_paths:
        gains_db = gains_db + [gains_db[-1]] * (n_paths - len(gains_db))
    paths = []
    for idx in range(n_paths):
        paths.append(
            {
                "delay_offset": 0.0 if idx == 0 else float(rng.uniform(0.0, delay_spread_s)),
                "amplitude": float(np.sqrt(db_to_linear(gains_db[idx]))),
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                "base_delay_s": float(base_delay_s),
            }
        )
    return paths


def build_indoor_channel_scenario(rician_k: float | None = None, use_rayleigh: bool = False, realizations: int = 1) -> dict[str, Any]:
    return {
        "rician_k": rician_k,
        "use_rayleigh": use_rayleigh,
        "realizations": realizations,
    }
