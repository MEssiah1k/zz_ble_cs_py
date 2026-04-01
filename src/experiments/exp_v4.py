"""V4 concurrent multi-device ranging with target matching."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..channel_models import add_awgn
from ..estimators import estimate_distance_by_phase_slope, estimate_distance_by_target_scan
from ..io_utils import save_dataframe
from ..metrics import summarize_errors
from ..monte_carlo import collect_trial_results, run_monte_carlo
from ..plotting import plot_distance_gap_vs_success, plot_num_devices_vs_error, plot_power_gap_vs_error, plot_score_curve
from ..signal_models import multi_link_frequency_response
from ..utils import db_to_linear, set_random_seed
from .common import config_distance_grid, finish_experiment, setup_experiment


def _build_device_distances(base_target_distance: float, n_devices: int, gap: float, target_index: int) -> np.ndarray:
    distances = np.array([base_target_distance + idx * gap for idx in range(n_devices)], dtype=float)
    distances[target_index] = base_target_distance
    return distances


def _build_amplitudes(n_devices: int, target_index: int, power_gap_db: float) -> np.ndarray:
    amps = np.ones(n_devices, dtype=float)
    interferer_amp = np.sqrt(db_to_linear(-power_gap_db))
    for idx in range(n_devices):
        if idx != target_index:
            amps[idx] = interferer_amp
    return amps


def _summarize_with_baselines(df: pd.DataFrame, success_threshold: float) -> dict:
    scan_summary = summarize_errors(df["true_distance"], df["est_distance"], thresholds=[success_threshold])
    legacy_summary = summarize_errors(df["true_distance"], df["legacy_est_distance"], thresholds=[success_threshold])
    naive_summary = summarize_errors(df["true_distance"], df["naive_est_distance"], thresholds=[success_threshold])
    return {
        **scan_summary,
        "legacy_scan_rmse": legacy_summary["rmse"],
        "legacy_scan_success_rate": legacy_summary[f"success_rate_le_{success_threshold}m"],
        "naive_rmse": naive_summary["rmse"],
        "naive_success_rate": naive_summary[f"success_rate_le_{success_threshold}m"],
        "mean_peak_ratio": float(df["peak_ratio"].mean()),
        "mean_peak_margin": float(df["peak_margin"].mean()),
        "mean_confidence": float(df["confidence"].mean()),
    }


def _trial(
    seed: int,
    freqs: np.ndarray,
    distance_grid: np.ndarray,
    true_distance: float,
    distances: np.ndarray,
    amplitudes: np.ndarray,
    target_index: int,
    snr_db: float,
    round_trip: bool,
    target_prior_sigma_m: float | None,
) -> dict:
    rng = set_random_seed(seed)
    response = multi_link_frequency_response(
        freqs,
        distances,
        amplitudes,
        phase_offsets=rng.uniform(0.0, 2.0 * np.pi, size=len(distances)),
        round_trip=round_trip,
    )
    noisy = add_awgn(response, snr_db=snr_db, rng=rng)
    target_scan = estimate_distance_by_target_scan(
        noisy,
        freqs,
        distance_grid,
        round_trip=round_trip,
        score_mode="composite",
        prior_distance=true_distance,
        prior_sigma_m=target_prior_sigma_m,
    )
    legacy_scan = estimate_distance_by_target_scan(noisy, freqs, distance_grid, round_trip=round_trip, score_mode="legacy")
    naive = estimate_distance_by_phase_slope(noisy, freqs, round_trip=round_trip)
    return {
        "trial_id": seed,
        "true_distance": true_distance,
        "est_distance": target_scan["distance_est"],
        "legacy_est_distance": legacy_scan["distance_est"],
        "naive_est_distance": naive["distance_est"],
        "abs_error": abs(target_scan["distance_est"] - true_distance),
        "legacy_abs_error": abs(legacy_scan["distance_est"] - true_distance),
        "naive_abs_error": abs(naive["distance_est"] - true_distance),
        "best_score": target_scan["best_score"],
        "second_best_score": target_scan["second_best_score"],
        "peak_margin": target_scan["peak_margin"],
        "peak_ratio": target_scan["peak_ratio"],
        "confidence": target_scan["confidence"],
        "n_devices": len(distances),
        "target_index": target_index,
        "seed": seed,
    }


def run(config: dict, overwrite: bool = False) -> dict:
    freqs, output_dir = setup_experiment(config, overwrite=overwrite)
    distance_grid = config_distance_grid(config)
    success_threshold = config["success_threshold_m"]

    trial_tables = []
    num_device_rows = []
    for n_devices in config["num_devices_list"]:
        distances = _build_device_distances(config["base_target_distance"], n_devices, gap=2.0, target_index=config["target_device_index"])
        amplitudes = np.ones(n_devices)
        results = run_monte_carlo(
            lambda seed: _trial(
                seed,
                freqs,
                distance_grid,
                config["base_target_distance"],
                distances,
                amplitudes,
                config["target_device_index"],
                config["snr_db"],
                config["round_trip"],
                config.get("target_prior_sigma_m"),
            ),
            config["monte_carlo_trials"],
            config["random_seed"] + n_devices * 97,
        )
        df = collect_trial_results(results)
        trial_tables.append(df.assign(scan_type="num_devices", scenario_value=n_devices))
        num_device_rows.append({"n_devices": n_devices, **_summarize_with_baselines(df, success_threshold)})

    distance_gap_rows = []
    for gap in config["distance_gap_list"]:
        distances = _build_device_distances(config["base_target_distance"], max(config["num_devices_list"]), gap=gap, target_index=config["target_device_index"])
        amplitudes = np.ones(len(distances))
        results = run_monte_carlo(
            lambda seed: _trial(
                seed,
                freqs,
                distance_grid,
                config["base_target_distance"],
                distances,
                amplitudes,
                config["target_device_index"],
                config["snr_db"],
                config["round_trip"],
                config.get("target_prior_sigma_m"),
            ),
            config["monte_carlo_trials"],
            config["random_seed"] + int(gap * 131),
        )
        df = collect_trial_results(results)
        trial_tables.append(df.assign(scan_type="distance_gap", scenario_value=gap))
        distance_gap_rows.append({"distance_gap_m": gap, **_summarize_with_baselines(df, success_threshold)})

    power_gap_rows = []
    for power_gap_db in config["power_gap_db_list"]:
        distances = _build_device_distances(config["base_target_distance"], max(config["num_devices_list"]), gap=2.0, target_index=config["target_device_index"])
        amplitudes = _build_amplitudes(len(distances), config["target_device_index"], power_gap_db)
        results = run_monte_carlo(
            lambda seed: _trial(
                seed,
                freqs,
                distance_grid,
                config["base_target_distance"],
                distances,
                amplitudes,
                config["target_device_index"],
                config["snr_db"],
                config["round_trip"],
                config.get("target_prior_sigma_m"),
            ),
            config["monte_carlo_trials"],
            config["random_seed"] + int((power_gap_db + 20) * 173),
        )
        df = collect_trial_results(results)
        trial_tables.append(df.assign(scan_type="power_gap", scenario_value=power_gap_db))
        power_gap_rows.append({"power_gap_db": power_gap_db, **_summarize_with_baselines(df, success_threshold)})

    example_distances = _build_device_distances(config["base_target_distance"], max(config["num_devices_list"]), gap=2.0, target_index=config["target_device_index"])
    example_amplitudes = _build_amplitudes(len(example_distances), config["target_device_index"], 3.0)
    example_response = multi_link_frequency_response(
        freqs,
        example_distances,
        example_amplitudes,
        phase_offsets=np.zeros(len(example_distances)),
        round_trip=config["round_trip"],
    )
    example_scan = estimate_distance_by_target_scan(
        example_response,
        freqs,
        distance_grid,
        round_trip=config["round_trip"],
        score_mode="composite",
        prior_distance=config["base_target_distance"],
        prior_sigma_m=config.get("target_prior_sigma_m"),
    )

    num_device_df = pd.DataFrame(num_device_rows)
    distance_gap_df = pd.DataFrame(distance_gap_rows)
    power_gap_df = pd.DataFrame(power_gap_rows)
    trial_df = pd.concat(trial_tables, ignore_index=True)

    save_dataframe(trial_df, output_dir / "tables", "trial_results.csv")
    save_dataframe(num_device_df, output_dir / "tables", "summary_num_devices.csv")
    save_dataframe(distance_gap_df, output_dir / "tables", "summary_distance_gap.csv")
    save_dataframe(power_gap_df, output_dir / "tables", "summary_power_gap.csv")

    plot_num_devices_vs_error(num_device_df["n_devices"].to_numpy(), num_device_df["rmse"].to_numpy(), output_dir / "figures" / "num_devices_vs_rmse.png")
    plot_distance_gap_vs_success(
        distance_gap_df["distance_gap_m"].to_numpy(),
        distance_gap_df[f"success_rate_le_{success_threshold}m"].to_numpy(),
        output_dir / "figures" / "distance_gap_vs_success.png",
    )
    plot_power_gap_vs_error(power_gap_df["power_gap_db"].to_numpy(), power_gap_df["rmse"].to_numpy(), output_dir / "figures" / "power_gap_vs_rmse.png")
    plot_score_curve(example_scan["distance_grid"], example_scan["scores"], output_dir / "figures" / "target_score_curve.png")

    return finish_experiment(
        {
            "n_trials": int(len(trial_df)),
            "best_num_device_rmse": float(num_device_df["rmse"].min()),
            "best_distance_gap_success": float(distance_gap_df[f"success_rate_le_{success_threshold}m"].max()),
            "best_power_gap_success": float(power_gap_df[f"success_rate_le_{success_threshold}m"].max()),
        },
        output_dir,
    )
