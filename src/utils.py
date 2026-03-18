"""Shared utility helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def db_to_linear(value_db: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(value_db) / 10.0)


def linear_to_db(value: float | np.ndarray, floor: float = 1e-15) -> float | np.ndarray:
    safe_value = np.maximum(np.asarray(value), floor)
    return 10.0 * np.log10(safe_value)


def set_random_seed(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def complex_awgn(shape: tuple[int, ...], snr_db: float, signal_power: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise_power = signal_power / db_to_linear(snr_db)
    sigma = np.sqrt(noise_power / 2.0)
    return sigma * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def normalize_complex_signal(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128)
    scale = np.maximum(np.abs(signal).max(axis=-1, keepdims=True), eps)
    return signal / scale


def unwrap_phase_safe(phase: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.unwrap(np.asarray(phase), axis=axis)


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, ensure_ascii=True)


def maybe_repeat(value: float | Iterable[float], count: int) -> np.ndarray:
    if np.isscalar(value):
        return np.full(count, float(value))
    arr = np.asarray(list(value), dtype=float)
    if arr.shape[0] != count:
        raise ValueError(f"Expected {count} values, got {arr.shape[0]}")
    return arr
