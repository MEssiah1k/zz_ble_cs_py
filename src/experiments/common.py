"""Shared experiment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..frequency_plan import build_frequency_grid
from ..io_utils import prepare_output_dir, save_config_snapshot, save_summary
from ..utils import ensure_dir


def config_distance_grid(config: dict[str, Any]) -> np.ndarray:
    grid = config["distance_grid"]
    return np.arange(grid["start"], grid["stop"] + 0.5 * grid["step"], grid["step"], dtype=float)


def setup_experiment(config: dict[str, Any], overwrite: bool = False) -> tuple[np.ndarray, Path]:
    freqs = build_frequency_grid(config["f_start"], config["f_step"], config["n_freqs"])
    output_dir = prepare_output_dir(config["save_dir"], config["version_name"], overwrite=overwrite)
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "tables")
    save_config_snapshot(config, output_dir)
    return freqs, output_dir


def finish_experiment(summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    save_summary(summary, output_dir)
    return summary


def resolve_prior_modes(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    prior_modes = config.get("prior_modes")
    if prior_modes:
        return prior_modes
    # 兼容旧配置；新评估应显式提供 prior_modes。
    sigma = config.get("target_prior_sigma_m", 0.75)
    return {
        "accurate_prior": {"bias_m": 0.0, "sigma_m": sigma},
    }


def build_prior_for_mode(
    prior_mode: str,
    prior_config: dict[str, Any],
    true_distance: float,
    distance_grid: np.ndarray,
) -> tuple[float | None, float | None]:
    if prior_mode == "no_prior":
        return None, None
    sigma_m = prior_config.get("sigma_m")
    if sigma_m is None:
        return None, None
    bias_m = float(prior_config.get("bias_m", 0.0))
    prior_distance = float(true_distance + bias_m)
    return float(np.clip(prior_distance, distance_grid.min(), distance_grid.max())), float(sigma_m)
