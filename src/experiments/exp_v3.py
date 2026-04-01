"""V3 repeated measurements with random phase offsets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..channel_models import add_awgn
from ..estimators import estimate_distance_by_phase_slope
from ..io_utils import save_dataframe
from ..metrics import summarize_errors
from ..monte_carlo import collect_trial_results, run_monte_carlo
from ..plotting import plot_boxplot_errors, plot_histogram_errors, plot_repeats_vs_error
from ..signal_models import repeated_measurements_single_link
from ..utils import set_random_seed
from .common import finish_experiment, setup_experiment


def _legacy_complex_average(responses: np.ndarray, freqs: np.ndarray, round_trip: bool) -> float:
    normalized = responses / (np.abs(responses) + 1e-12)
    aggregated = np.mean(normalized, axis=0)
    return estimate_distance_by_phase_slope(aggregated, freqs, round_trip=round_trip)["distance_est"]


def _reference_aligned_average(responses: np.ndarray, freqs: np.ndarray, round_trip: bool) -> float:
    # 先用首频点相位对齐每次测量的常量偏置，再做相干聚合。
    reference = responses[:, :1] / (np.abs(responses[:, :1]) + 1e-12)
    aligned = responses * np.conj(reference)
    normalized = aligned / (np.abs(aligned) + 1e-12)
    aggregated = np.mean(normalized, axis=0)
    return estimate_distance_by_phase_slope(aggregated, freqs, round_trip=round_trip)["distance_est"]


def _slope_fusion_average(responses: np.ndarray, freqs: np.ndarray, round_trip: bool) -> float:
    per_repeat = [
        estimate_distance_by_phase_slope(response, freqs, round_trip=round_trip)["distance_est"]
        for response in responses
    ]
    return float(np.median(per_repeat))


def run(config: dict, overwrite: bool = False) -> dict:
    freqs, output_dir = setup_experiment(config, overwrite=overwrite)
    trial_rows = []
    summary_rows = []
    success_threshold = config["success_threshold_m"]
    for n_repeats in config["n_repeats_list"]:

        def trial_fn(seed: int) -> dict:
            rng = set_random_seed(seed)
            responses = repeated_measurements_single_link(
                freqs,
                config["true_distance"],
                n_repeats=n_repeats,
                random_phase=config["random_phase_enable"],
                round_trip=config["round_trip"],
                rng=rng,
            )
            if "snr_db" in config:
                responses = np.asarray([add_awgn(response, snr_db=config["snr_db"], rng=rng) for response in responses])
            single_est = estimate_distance_by_phase_slope(responses[0], freqs, round_trip=config["round_trip"])["distance_est"]
            legacy_est = _legacy_complex_average(responses, freqs, config["round_trip"])
            aligned_est = _reference_aligned_average(responses, freqs, config["round_trip"])
            fused_est = _slope_fusion_average(responses, freqs, config["round_trip"])
            return {
                "trial_id": seed,
                "true_distance": config["true_distance"],
                "single_est_distance": single_est,
                "legacy_est_distance": legacy_est,
                "aligned_est_distance": aligned_est,
                "fused_est_distance": fused_est,
                "single_abs_error": abs(single_est - config["true_distance"]),
                "legacy_abs_error": abs(legacy_est - config["true_distance"]),
                "aligned_abs_error": abs(aligned_est - config["true_distance"]),
                "fused_abs_error": abs(fused_est - config["true_distance"]),
                "n_repeats": n_repeats,
                "seed": seed,
            }

        results = run_monte_carlo(trial_fn, config["monte_carlo_trials"], config["random_seed"] + n_repeats * 17)
        df = collect_trial_results(results)
        trial_rows.append(df)
        legacy_summary = summarize_errors(df["true_distance"], df["legacy_est_distance"], thresholds=[success_threshold])
        aligned_summary = summarize_errors(df["true_distance"], df["aligned_est_distance"], thresholds=[success_threshold])
        fused_summary = summarize_errors(df["true_distance"], df["fused_est_distance"], thresholds=[success_threshold])
        single_summary = summarize_errors(df["true_distance"], df["single_est_distance"], thresholds=[success_threshold])
        summary_rows.append(
            {
                "n_repeats": n_repeats,
                "legacy_rmse": legacy_summary["rmse"],
                "legacy_success_rate": legacy_summary[f"success_rate_le_{success_threshold}m"],
                "aligned_rmse": aligned_summary["rmse"],
                "aligned_success_rate": aligned_summary[f"success_rate_le_{success_threshold}m"],
                "fused_rmse": fused_summary["rmse"],
                "fused_success_rate": fused_summary[f"success_rate_le_{success_threshold}m"],
                "single_rmse": single_summary["rmse"],
                "single_success_rate": single_summary[f"success_rate_le_{success_threshold}m"],
            }
        )

    trial_df = pd.concat(trial_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(trial_df, output_dir / "tables", "trial_results.csv")
    save_dataframe(summary_df, output_dir / "tables", "summary_by_repeats.csv")
    plot_repeats_vs_error(summary_df["n_repeats"].to_numpy(), summary_df["aligned_rmse"].to_numpy(), output_dir / "figures" / "repeats_vs_rmse.png")
    plot_histogram_errors(trial_df["aligned_abs_error"].to_numpy(), output_dir / "figures" / "aggregate_error_histogram.png")
    grouped_errors = [trial_df[trial_df["n_repeats"] == repeat]["aligned_abs_error"].to_numpy() for repeat in config["n_repeats_list"]]
    plot_boxplot_errors(grouped_errors, [str(val) for val in config["n_repeats_list"]], output_dir / "figures" / "error_boxplot.png")
    return finish_experiment(
        {
            "n_trials": int(len(trial_df)),
            "best_aligned_rmse": float(summary_df["aligned_rmse"].min()),
            "best_fused_rmse": float(summary_df["fused_rmse"].min()),
        },
        output_dir,
    )
