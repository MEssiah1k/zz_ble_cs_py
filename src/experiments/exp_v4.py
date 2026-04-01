"""V4 concurrent ranging with peer-IQ matched multifrequency suppression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..channel_models import add_awgn
from ..estimators import estimate_distance_by_peer_multifreq, estimate_distance_by_phase_slope, estimate_distance_by_target_scan
from ..io_utils import save_dataframe
from ..metrics import summarize_errors
from ..plotting import plot_num_devices_vs_error, plot_power_gap_vs_error, plot_score_curve
from ..scenarios import build_random_ranging_scenario
from ..signal_models import build_paired_iq_multi_link
from ..utils import db_to_linear, set_random_seed
from .common import config_distance_grid, finish_experiment, setup_experiment


def _build_amplitudes(n_devices: int, target_index: int, power_gap_db: float) -> np.ndarray:
    amplitudes = np.ones(n_devices, dtype=float)
    interferer_amp = np.sqrt(db_to_linear(-power_gap_db))
    for idx in range(n_devices):
        if idx != target_index:
            amplitudes[idx] = interferer_amp
    return amplitudes


def _encode_distances(distances: np.ndarray) -> str:
    return ";".join(f"{distance:.3f}" for distance in np.asarray(distances, dtype=float))


def _summarize_group(df: pd.DataFrame, success_threshold: float) -> dict:
    peer_summary = summarize_errors(df["true_distance"], df["est_distance"], thresholds=[success_threshold])
    raw_scan_summary = summarize_errors(df["true_distance"], df["legacy_est_distance"], thresholds=[success_threshold])
    naive_summary = summarize_errors(df["true_distance"], df["naive_est_distance"], thresholds=[success_threshold])
    peer_slope_summary = summarize_errors(df["true_distance"], df["peer_slope_est_distance"], thresholds=[success_threshold])
    return {
        **peer_summary,
        "raw_scan_rmse": raw_scan_summary["rmse"],
        "raw_scan_success_rate": raw_scan_summary[f"success_rate_le_{success_threshold}m"],
        "naive_rmse": naive_summary["rmse"],
        "naive_success_rate": naive_summary[f"success_rate_le_{success_threshold}m"],
        "peer_slope_rmse": peer_slope_summary["rmse"],
        "peer_slope_success_rate": peer_slope_summary[f"success_rate_le_{success_threshold}m"],
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


def _neighbor_gap_summary(df: pd.DataFrame, bins: list[float], success_threshold: float) -> pd.DataFrame:
    work_df = df.copy()
    labels = [f"[{bins[idx]}, {bins[idx + 1]})" for idx in range(len(bins) - 1)]
    work_df["neighbor_gap_bin"] = pd.cut(work_df["target_neighbor_gap"], bins=bins, labels=labels, include_lowest=True, right=False)
    work_df = work_df.dropna(subset=["neighbor_gap_bin"])
    return _group_summary(work_df, ["neighbor_gap_bin"], success_threshold)


def _scenario_row(
    seed: int,
    freqs: np.ndarray,
    distance_grid: np.ndarray,
    config: dict,
    n_devices: int,
    power_gap_db: float,
    scan_type: str,
    scenario_value: float,
) -> dict:
    rng = set_random_seed(seed)
    scenario = build_random_ranging_scenario(
        num_devices=n_devices,
        target_distance_range_m=tuple(config["target_distance_range_m"]),
        device_distance_range_m=tuple(config["device_distance_range_m"]),
        min_device_spacing_m=float(config["min_device_spacing_m"]),
        rng=rng,
        target_index=None if config.get("random_target_index", True) else config.get("target_device_index", 0),
    )
    phase_offsets = rng.uniform(0.0, 2.0 * np.pi, size=(n_devices, len(freqs)))
    amplitudes = _build_amplitudes(n_devices, scenario["target_index"], power_gap_db)
    local_iq, peer_iqs = build_paired_iq_multi_link(freqs, scenario["distances"], amplitudes, phase_offsets=phase_offsets, round_trip=config["round_trip"])
    noisy_local = add_awgn(local_iq, snr_db=config["snr_db"], rng=rng)
    target_peer = peer_iqs[scenario["target_index"]]

    estimate = estimate_distance_by_peer_multifreq(noisy_local, target_peer, freqs, distance_grid, round_trip=config["round_trip"])
    raw_scan = estimate_distance_by_target_scan(noisy_local, freqs, distance_grid, round_trip=config["round_trip"], score_mode="legacy")
    naive = estimate_distance_by_phase_slope(noisy_local, freqs, round_trip=config["round_trip"])
    peer_slope = estimate_distance_by_phase_slope(estimate["matched_response"], freqs, round_trip=config["round_trip"])

    return {
        "trial_id": seed,
        "scan_type": scan_type,
        "scenario_value": scenario_value,
        "true_distance": scenario["target_distance"],
        "target_distance": scenario["target_distance"],
        "target_index": scenario["target_index"],
        "num_devices": n_devices,
        "power_gap_db": power_gap_db,
        "device_distances": _encode_distances(scenario["distances"]),
        "mean_neighbor_gap": scenario["mean_neighbor_gap"],
        "min_neighbor_gap": scenario["min_neighbor_gap"],
        "target_neighbor_gap": scenario["target_neighbor_gap"],
        "est_distance": estimate["distance_est"],
        "legacy_est_distance": raw_scan["distance_est"],
        "naive_est_distance": naive["distance_est"],
        "peer_slope_est_distance": peer_slope["distance_est"],
        "abs_error": abs(estimate["distance_est"] - scenario["target_distance"]),
        "legacy_abs_error": abs(raw_scan["distance_est"] - scenario["target_distance"]),
        "naive_abs_error": abs(naive["distance_est"] - scenario["target_distance"]),
        "peer_slope_abs_error": abs(peer_slope["distance_est"] - scenario["target_distance"]),
        "best_score": estimate["best_score"],
        "second_best_score": estimate["second_best_score"],
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
    for n_devices in config["num_devices_list"]:
        seed_base = config["random_seed"] + n_devices * 97
        for trial_idx in range(config["monte_carlo_trials"]):
            trial_rows.append(
                _scenario_row(
                    seed_base + trial_idx,
                    freqs,
                    distance_grid,
                    config,
                    n_devices=n_devices,
                    power_gap_db=0.0,
                    scan_type="num_devices",
                    scenario_value=float(n_devices),
                )
            )
    for power_gap_db in config["power_gap_db_list"]:
        seed_base = config["random_seed"] + int((power_gap_db + 20.0) * 173)
        for trial_idx in range(config["monte_carlo_trials"]):
            trial_rows.append(
                _scenario_row(
                    seed_base + trial_idx,
                    freqs,
                    distance_grid,
                    config,
                    n_devices=int(config.get("power_gap_num_devices", max(config["num_devices_list"]))),
                    power_gap_db=float(power_gap_db),
                    scan_type="power_gap",
                    scenario_value=float(power_gap_db),
                )
            )

    trial_df = pd.DataFrame(trial_rows)
    num_device_df = _group_summary(trial_df[trial_df["scan_type"] == "num_devices"], ["num_devices"], success_threshold)
    power_gap_df = _group_summary(trial_df[trial_df["scan_type"] == "power_gap"], ["power_gap_db"], success_threshold)
    overall_df = pd.DataFrame([_summarize_group(trial_df, success_threshold)])
    neighbor_gap_df = _neighbor_gap_summary(trial_df[trial_df["scan_type"] == "num_devices"], config["neighbor_gap_bins_m"], success_threshold)

    example_row = trial_df.iloc[0]
    example_rng = set_random_seed(int(example_row["seed"]))
    example_scenario = build_random_ranging_scenario(
        num_devices=int(example_row["num_devices"]),
        target_distance_range_m=tuple(config["target_distance_range_m"]),
        device_distance_range_m=tuple(config["device_distance_range_m"]),
        min_device_spacing_m=float(config["min_device_spacing_m"]),
        rng=example_rng,
        target_index=None if config.get("random_target_index", True) else config.get("target_device_index", 0),
    )
    example_phase_offsets = example_rng.uniform(0.0, 2.0 * np.pi, size=(int(example_row["num_devices"]), len(freqs)))
    example_amps = _build_amplitudes(int(example_row["num_devices"]), example_scenario["target_index"], float(example_row["power_gap_db"]))
    example_local, example_peers = build_paired_iq_multi_link(freqs, example_scenario["distances"], example_amps, phase_offsets=example_phase_offsets, round_trip=config["round_trip"])
    example_noisy = add_awgn(example_local, snr_db=config["snr_db"], rng=example_rng)
    example_estimate = estimate_distance_by_peer_multifreq(example_noisy, example_peers[example_scenario["target_index"]], freqs, distance_grid, round_trip=config["round_trip"])

    save_dataframe(trial_df, output_dir / "tables", "trial_results.csv")
    save_dataframe(num_device_df, output_dir / "tables", "summary_num_devices.csv")
    save_dataframe(power_gap_df, output_dir / "tables", "summary_power_gap.csv")
    save_dataframe(overall_df, output_dir / "tables", "summary_overall.csv")
    save_dataframe(neighbor_gap_df, output_dir / "tables", "summary_neighbor_gap.csv")

    plot_num_devices_vs_error(num_device_df["num_devices"].to_numpy(), num_device_df["rmse"].to_numpy(), output_dir / "figures" / "num_devices_vs_rmse.png")
    plot_power_gap_vs_error(power_gap_df["power_gap_db"].to_numpy(), power_gap_df["rmse"].to_numpy(), output_dir / "figures" / "power_gap_vs_rmse.png")
    plot_score_curve(example_estimate["distance_grid"], example_estimate["scores"], output_dir / "figures" / "target_score_curve.png")

    summary = {
        "n_trials": int(len(trial_df)),
        "peer_multifreq_rmse": float(overall_df.loc[0, "rmse"]),
        f"peer_multifreq_success_rate_le_{success_threshold}m": float(overall_df.loc[0, f"success_rate_le_{success_threshold}m"]),
        "raw_scan_rmse": float(overall_df.loc[0, "raw_scan_rmse"]),
        "naive_rmse": float(overall_df.loc[0, "naive_rmse"]),
    }
    return finish_experiment(summary, output_dir)
