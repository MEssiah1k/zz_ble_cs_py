"""V5 multipath experiment on top of concurrent ranging."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..channel_models import add_awgn, apply_single_reflector_multipath
from ..estimators import estimate_distance_by_target_scan
from ..io_utils import save_dataframe
from ..metrics import summarize_errors
from ..monte_carlo import collect_trial_results, run_monte_carlo
from ..plotting import plot_multipath_vs_error
from ..utils import db_to_linear, set_random_seed
from .common import config_distance_grid, finish_experiment, setup_experiment


def _build_response(freqs: np.ndarray, target_distance: float, n_devices: int, n_paths: int, path_gain_db: float, delay_spread_s: float, rng: np.random.Generator) -> np.ndarray:
    response = np.zeros(len(freqs), dtype=np.complex128)
    for idx in range(n_devices):
        distance = target_distance + idx * 2.0
        extra_paths = []
        for _ in range(max(n_paths - 1, 0)):
            extra_paths.append({"delay_offset": float(rng.uniform(0.0, delay_spread_s)), "amplitude": float(np.sqrt(db_to_linear(path_gain_db))), "phase": float(rng.uniform(0.0, 2.0 * np.pi))})
        response += apply_single_reflector_multipath(freqs, distance, extra_paths)
    return response


def run(config: dict, overwrite: bool = False) -> dict:
    freqs, output_dir = setup_experiment(config, overwrite=overwrite)
    distance_grid = config_distance_grid(config)
    n_path_rows = []
    power_rows = []

    for n_paths in config["n_paths_list"]:
        results = run_monte_carlo(
            lambda seed: _trial(seed, freqs, distance_grid, config, n_paths, config["path_gain_db_list"][0]),
            config["monte_carlo_trials"],
            config["random_seed"] + n_paths * 31,
        )
        df = collect_trial_results(results)
        metrics = summarize_errors(df["true_distance"], df["est_distance"], thresholds=[config["success_threshold_m"]])
        n_path_rows.append({"n_paths": n_paths, **metrics})

    for path_gain_db in config["path_gain_db_list"]:
        results = run_monte_carlo(
            lambda seed: _trial(seed, freqs, distance_grid, config, config["n_paths_list"][-1], path_gain_db),
            config["monte_carlo_trials"],
            config["random_seed"] + int(abs(path_gain_db) * 43),
        )
        df = collect_trial_results(results)
        metrics = summarize_errors(df["true_distance"], df["est_distance"], thresholds=[config["success_threshold_m"]])
        power_rows.append({"path_gain_db": path_gain_db, **metrics})

    n_path_df = pd.DataFrame(n_path_rows)
    power_df = pd.DataFrame(power_rows)
    save_dataframe(n_path_df, output_dir / "tables", "summary_n_paths.csv")
    save_dataframe(power_df, output_dir / "tables", "summary_path_gain.csv")
    plot_multipath_vs_error(n_path_df["n_paths"].to_numpy(), n_path_df["rmse"].to_numpy(), output_dir / "figures" / "n_paths_vs_rmse.png", xlabel="Number of paths")
    plot_multipath_vs_error(power_df["path_gain_db"].to_numpy(), power_df["rmse"].to_numpy(), output_dir / "figures" / "path_gain_vs_rmse.png", xlabel="Reflected path gain (dB)")
    return finish_experiment({"best_rmse": float(min(n_path_df["rmse"].min(), power_df["rmse"].min()))}, output_dir)


def _trial(seed: int, freqs: np.ndarray, distance_grid: np.ndarray, config: dict, n_paths: int, path_gain_db: float) -> dict:
    rng = set_random_seed(seed)
    response = _build_response(freqs, config["base_target_distance"], config["num_devices"], n_paths, path_gain_db, config["delay_spread_s"], rng)
    noisy = add_awgn(response, config["snr_db"], rng=rng)
    estimate = estimate_distance_by_target_scan(noisy, freqs, distance_grid, round_trip=config["round_trip"])
    return {
        "trial_id": seed,
        "true_distance": config["base_target_distance"],
        "est_distance": estimate["distance_est"],
        "abs_error": abs(estimate["distance_est"] - config["base_target_distance"]),
        "n_paths": n_paths,
        "path_gain_db": path_gain_db,
        "seed": seed,
    }
