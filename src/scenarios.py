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


def _distance_candidates(value_range: tuple[float, float], min_spacing_m: float) -> np.ndarray:
    low, high = float(value_range[0]), float(value_range[1])
    if low >= high:
        raise ValueError("distance range must satisfy low < high")
    if min_spacing_m <= 0.0:
        raise ValueError("min_spacing_m must be positive")
    return np.arange(low, high + 0.5 * min_spacing_m, min_spacing_m, dtype=float)


def build_random_ranging_scenario(
    num_devices: int,
    target_distance_range_m: tuple[float, float],
    device_distance_range_m: tuple[float, float],
    min_device_spacing_m: float,
    rng: np.random.Generator | None = None,
    target_index: int | None = None,
    amplitudes: float | list[float] = 1.0,
    random_phase: bool = True,
) -> dict[str, Any]:
    rng = rng or np.random.default_rng()
    if num_devices < 1:
        raise ValueError("num_devices must be at least 1")
    resolved_target_index = int(rng.integers(0, num_devices)) if target_index is None else int(target_index)
    if not 0 <= resolved_target_index < num_devices:
        raise ValueError("target_index out of range")

    target_candidates = _distance_candidates(target_distance_range_m, min_device_spacing_m)
    device_candidates = _distance_candidates(device_distance_range_m, min_device_spacing_m)
    if target_candidates.size == 0 or device_candidates.size == 0:
        raise ValueError("No valid distance candidates available")

    target_distance = float(rng.choice(target_candidates))
    available_interferers = device_candidates[np.abs(device_candidates - target_distance) >= min_device_spacing_m]
    if available_interferers.size < num_devices - 1:
        raise ValueError("Not enough device distance candidates for the requested spacing constraint")
    interferer_distances = rng.choice(available_interferers, size=num_devices - 1, replace=False).astype(float).tolist()
    distances = list(interferer_distances)
    distances.insert(resolved_target_index, target_distance)
    distances_arr = np.asarray(distances, dtype=float)
    sorted_distances = np.sort(distances_arr)
    adjacent_gaps = np.diff(sorted_distances) if len(sorted_distances) > 1 else np.array([], dtype=float)
    target_neighbor_gap = float(np.min(np.abs(np.delete(distances_arr, resolved_target_index) - target_distance))) if num_devices > 1 else np.nan
    phases = rng.uniform(0.0, 2.0 * np.pi, size=num_devices) if random_phase else np.zeros(num_devices)
    return {
        "distances": distances_arr,
        "target_distance": float(target_distance),
        "target_index": resolved_target_index,
        "amplitudes": maybe_repeat(amplitudes, num_devices),
        "phase_offsets": phases,
        "mean_neighbor_gap": float(adjacent_gaps.mean()) if adjacent_gaps.size else np.nan,
        "min_neighbor_gap": float(adjacent_gaps.min()) if adjacent_gaps.size else np.nan,
        "target_neighbor_gap": target_neighbor_gap,
    }
