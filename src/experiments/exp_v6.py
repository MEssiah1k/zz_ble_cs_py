"""V6 indoor fading experiment using peer-IQ matched multifrequency suppression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..channel_models import add_awgn, apply_rayleigh_fading, apply_rician_fading
from ..estimators import estimate_distance_by_peer_multifreq, estimate_distance_by_phase_slope, estimate_distance_by_target_scan
from ..io_utils import save_dataframe
from ..metrics import summarize_errors
from ..plotting import plot_histogram_errors, plot_kfactor_vs_error
from ..scenarios import build_random_ranging_scenario
from ..signal_models import single_link_frequency_response
from ..utils import set_random_seed
from .common import config_distance_grid, finish_experiment, setup_experiment


def _encode_distances(distances: np.ndarray) -> str:
    return ";".join(f"{distance:.3f}" for distance in np.asarray(distances, dtype=float))


def _build_local_and_peer(
    freqs: np.ndarray,
    distances: np.ndarray,
    phase_offsets: np.ndarray,
    round_trip: bool,
    target_index: int,
    k_factor: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    local_iq = np.zeros(len(freqs), dtype=np.complex128)
    peer_iqs = np.zeros((len(distances), len(freqs)), dtype=np.complex128)
    for dev_idx, distance in enumerate(np.asarray(distances, dtype=float)):
        response = single_link_frequency_response(freqs, float(distance), phase_offset=phase_offsets[dev_idx], round_trip=round_trip)
        if k_factor is None:
            response = apply_rayleigh_fading(response, rng=rng)
        else:
            response = apply_rician_fading(response, k_factor=k_factor, rng=rng)
        local_iq += response
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


def _scenario_row(seed: int, freqs: np.ndarray, distance_grid: np.ndarray, config: dict, k_factor: float | None) -> dict:
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
    local_iq, target_peer = _build_local_and_peer(freqs, scenario["distances"], phase_offsets, config["round_trip"], scenario["target_index"], k_factor, rng)
    noisy_local = add_awgn(local_iq, snr_db=config["snr_db"], rng=rng)
    estimate = estimate_distance_by_peer_multifreq(noisy_local, target_peer, freqs, distance_grid, round_trip=config["round_trip"])
    raw_scan = estimate_distance_by_target_scan(noisy_local, freqs, distance_grid, round_trip=config["round_trip"], score_mode="legacy")
    naive = estimate_distance_by_phase_slope(noisy_local, freqs, round_trip=config["round_trip"])
    return {
        "trial_id": seed,
        "true_distance": scenario["target_distance"],
        "target_distance": scenario["target_distance"],
        "target_index": scenario["target_index"],
        "num_devices": int(config["num_devices"]),
        "device_distances": _encode_distances(scenario["distances"]),
        "mean_neighbor_gap": scenario["mean_neighbor_gap"],
        "min_neighbor_gap": scenario["min_neighbor_gap"],
        "target_neighbor_gap": scenario["target_neighbor_gap"],
        "est_distance": estimate["distance_est"],
        "raw_scan_est_distance": raw_scan["distance_est"],
        "naive_est_distance": naive["distance_est"],
        "abs_error": abs(estimate["distance_est"] - scenario["target_distance"]),
        "peak_margin": estimate["peak_margin"],
        "peak_ratio": estimate["peak_ratio"],
        "confidence": estimate["confidence"],
        "k_factor": -1.0 if k_factor is None else float(k_factor),
        "channel": "rayleigh" if k_factor is None else "rician",
        "seed": seed,
    }


def run(config: dict, overwrite: bool = False) -> dict:
    freqs, output_dir = setup_experiment(config, overwrite=overwrite)
    distance_grid = config_distance_grid(config)
    success_threshold = config["success_threshold_m"]
    trial_rows: list[dict] = []

    for k_factor in config["rician_k_list"]:
        seed_base = config["random_seed"] + int(k_factor * 59)
        for trial_idx in range(config["channel_realizations"]):
            trial_rows.append(_scenario_row(seed_base + trial_idx, freqs, distance_grid, config, float(k_factor)))
    rayleigh_seed_base = config["random_seed"] + 9991
    for trial_idx in range(config["channel_realizations"]):
        trial_rows.append(_scenario_row(rayleigh_seed_base + trial_idx, freqs, distance_grid, config, None))

    trial_df = pd.DataFrame(trial_rows)
    k_df = _group_summary(trial_df[trial_df["channel"] == "rician"], ["k_factor"], success_threshold)
    channel_df = _group_summary(trial_df, ["channel"], success_threshold)

    save_dataframe(k_df, output_dir / "tables", "summary_k_factor.csv")
    save_dataframe(channel_df, output_dir / "tables", "summary_channels.csv")
    save_dataframe(trial_df, output_dir / "tables", "trial_results.csv")

    plot_kfactor_vs_error(k_df["k_factor"].to_numpy(), k_df["rmse"].to_numpy(), output_dir / "figures" / "kfactor_vs_rmse.png")
    plot_histogram_errors(trial_df["abs_error"].to_numpy(), output_dir / "figures" / "error_histogram.png")

    rician_row = channel_df[channel_df["channel"] == "rician"].iloc[0]
    summary = {
        "n_trials": int(len(trial_df)),
        "peer_multifreq_rmse": float(rician_row["rmse"]),
        f"peer_multifreq_success_rate_le_{success_threshold}m": float(rician_row[f"success_rate_le_{success_threshold}m"]),
        "raw_scan_rmse": float(rician_row["raw_scan_rmse"]),
        "naive_rmse": float(rician_row["naive_rmse"]),
    }
    return finish_experiment(summary, output_dir)
