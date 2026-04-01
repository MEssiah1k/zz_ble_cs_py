"""V5 multipath experiment using peer-IQ matched multifrequency suppression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..channel_models import add_awgn, apply_single_reflector_multipath
from ..estimators import estimate_distance_by_peer_multifreq, estimate_distance_by_phase_slope, estimate_distance_by_target_scan
from ..io_utils import save_dataframe
from ..metrics import summarize_errors
from ..plotting import plot_multipath_vs_error
from ..scenarios import build_random_ranging_scenario
from ..utils import db_to_linear, set_random_seed
from .common import config_distance_grid, finish_experiment, setup_experiment


def _encode_distances(distances: np.ndarray) -> str:
    return ";".join(f"{distance:.3f}" for distance in np.asarray(distances, dtype=float))


def _build_local_and_peer(
    freqs: np.ndarray,
    distances: np.ndarray,
    n_paths: int,
    path_gain_db: float,
    delay_spread_s: float,
    phase_offsets: np.ndarray,
    round_trip: bool,
    target_index: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    local_iq = np.zeros(len(freqs), dtype=np.complex128)
    peer_iqs = np.zeros((len(distances), len(freqs)), dtype=np.complex128)
    for dev_idx, distance in enumerate(np.asarray(distances, dtype=float)):
        extra_paths = []
        for _ in range(max(n_paths - 1, 0)):
            extra_paths.append(
                {
                    "delay_offset": float(rng.uniform(0.0, delay_spread_s)),
                    "amplitude": float(np.sqrt(db_to_linear(path_gain_db))),
                    "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                }
            )
        response = apply_single_reflector_multipath(freqs, float(distance), extra_paths)
        local_iq += response * np.exp(1j * phase_offsets[dev_idx])
        peer_iqs[dev_idx] = np.exp(1j * phase_offsets[dev_idx])
    return local_iq, peer_iqs[target_index]


def _summarize_group(df: pd.DataFrame, success_threshold: float) -> dict:
    peer_summary = summarize_errors(df["true_distance"], df["est_distance"], thresholds=[success_threshold])
    raw_scan_summary = summarize_errors(df["true_distance"], df["raw_scan_est_distance"], thresholds=[success_threshold])
    naive_summary = summarize_errors(df["true_distance"], df["naive_est_distance"], thresholds=[success_threshold])
    return {
        **peer_summary,
        "raw_scan_rmse": raw_scan_summary["rmse"],
        "naive_rmse": naive_summary["rmse"],
        "mean_peak_ratio": float(df["peak_ratio"].mean()),
        "mean_peak_margin": float(df["peak_margin"].mean()),
        "mean_confidence": float(df["confidence"].mean()),
        "mean_target_distance": float(df["target_distance"].mean()),
        "mean_neighbor_gap": float(df["mean_neighbor_gap"].mean()),
        "mean_target_neighbor_gap": float(df["target_neighbor_gap"].mean()),
        "n_trials": int(len(df)),
    }


def _group_summary(df: pd.DataFrame, group_cols: list[str], success_threshold: float) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(_summarize_group(group, success_threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def _scenario_row(seed: int, freqs: np.ndarray, distance_grid: np.ndarray, config: dict, n_paths: int, path_gain_db: float, scan_type: str, scenario_value: float) -> dict:
    rng = set_random_seed(seed)
    scenario = build_random_ranging_scenario(
        num_devices=int(config["num_devices"]),
        target_distance_range_m=tuple(config["target_distance_range_m"]),
        device_distance_range_m=tuple(config["device_distance_range_m"]),
        min_device_spacing_m=float(config["min_device_spacing_m"]),
        rng=rng,
        target_index=None if config.get("random_target_index", True) else config.get("target_device_index", 0),
    )
    phase_offsets = rng.uniform(0.0, 2.0 * np.pi, size=(int(config["num_devices"]), len(freqs)))
    local_iq, target_peer = _build_local_and_peer(
        freqs,
        scenario["distances"],
        n_paths,
        path_gain_db,
        config["delay_spread_s"],
        phase_offsets,
        config["round_trip"],
        scenario["target_index"],
        rng,
    )
    noisy_local = add_awgn(local_iq, config["snr_db"], rng=rng)
    estimate = estimate_distance_by_peer_multifreq(noisy_local, target_peer, freqs, distance_grid, round_trip=config["round_trip"])
    raw_scan = estimate_distance_by_target_scan(noisy_local, freqs, distance_grid, round_trip=config["round_trip"], score_mode="legacy")
    naive = estimate_distance_by_phase_slope(noisy_local, freqs, round_trip=config["round_trip"])
    return {
        "trial_id": seed,
        "scan_type": scan_type,
        "scenario_value": scenario_value,
        "true_distance": scenario["target_distance"],
        "target_distance": scenario["target_distance"],
        "target_index": scenario["target_index"],
        "num_devices": int(config["num_devices"]),
        "device_distances": _encode_distances(scenario["distances"]),
        "mean_neighbor_gap": scenario["mean_neighbor_gap"],
        "min_neighbor_gap": scenario["min_neighbor_gap"],
        "target_neighbor_gap": scenario["target_neighbor_gap"],
        "n_paths": n_paths,
        "path_gain_db": path_gain_db,
        "est_distance": estimate["distance_est"],
        "raw_scan_est_distance": raw_scan["distance_est"],
        "naive_est_distance": naive["distance_est"],
        "abs_error": abs(estimate["distance_est"] - scenario["target_distance"]),
        "peak_margin": estimate["peak_margin"],
        "peak_ratio": estimate["peak_ratio"],
        "confidence": estimate["confidence"],
        "seed": seed,
    }


def run(config: dict, overwrite: bool = False) -> dict:
    freqs, output_dir = setup_experiment(config, overwrite=overwrite)
    distance_grid = config_distance_grid(config)
    success_threshold = config["success_threshold_m"]
    trial_rows: list[dict] = []

    for n_paths in config["n_paths_list"]:
        seed_base = config["random_seed"] + n_paths * 31
        for trial_idx in range(config["monte_carlo_trials"]):
            trial_rows.append(_scenario_row(seed_base + trial_idx, freqs, distance_grid, config, n_paths, float(config["path_gain_db_list"][0]), "n_paths", float(n_paths)))
    for path_gain_db in config["path_gain_db_list"]:
        seed_base = config["random_seed"] + int(abs(path_gain_db) * 43)
        for trial_idx in range(config["monte_carlo_trials"]):
            trial_rows.append(_scenario_row(seed_base + trial_idx, freqs, distance_grid, config, int(config["n_paths_list"][-1]), float(path_gain_db), "path_gain", float(path_gain_db)))

    trial_df = pd.DataFrame(trial_rows)
    n_path_df = _group_summary(trial_df[trial_df["scan_type"] == "n_paths"], ["n_paths"], success_threshold)
    power_df = _group_summary(trial_df[trial_df["scan_type"] == "path_gain"], ["path_gain_db"], success_threshold)
    overall_df = pd.DataFrame([_summarize_group(trial_df, success_threshold)])

    save_dataframe(trial_df, output_dir / "tables", "trial_results.csv")
    save_dataframe(n_path_df, output_dir / "tables", "summary_n_paths.csv")
    save_dataframe(power_df, output_dir / "tables", "summary_path_gain.csv")
    save_dataframe(overall_df, output_dir / "tables", "summary_overall.csv")

    plot_multipath_vs_error(n_path_df["n_paths"].to_numpy(), n_path_df["rmse"].to_numpy(), output_dir / "figures" / "n_paths_vs_rmse.png", xlabel="Number of paths")
    plot_multipath_vs_error(power_df["path_gain_db"].to_numpy(), power_df["rmse"].to_numpy(), output_dir / "figures" / "path_gain_vs_rmse.png", xlabel="Reflected path gain (dB)")

    summary = {
        "n_trials": int(len(trial_df)),
        "peer_multifreq_rmse": float(overall_df.loc[0, "rmse"]),
        f"peer_multifreq_success_rate_le_{success_threshold}m": float(overall_df.loc[0, f"success_rate_le_{success_threshold}m"]),
        "raw_scan_rmse": float(overall_df.loc[0, "raw_scan_rmse"]),
        "naive_rmse": float(overall_df.loc[0, "naive_rmse"]),
    }
    return finish_experiment(summary, output_dir)
