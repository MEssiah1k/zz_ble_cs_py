"""V3 repeated measurements with random phase offsets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..estimators import estimate_distance_by_phase_slope
from ..io_utils import save_dataframe
from ..metrics import summarize_errors
from ..monte_carlo import collect_trial_results, run_monte_carlo
from ..plotting import plot_boxplot_errors, plot_histogram_errors, plot_repeats_vs_error
from ..signal_models import repeated_measurements_single_link
from ..utils import set_random_seed
from .common import finish_experiment, setup_experiment


def _estimate_from_repeats(responses: np.ndarray, freqs: np.ndarray, round_trip: bool) -> float:
    normalized = responses / (np.abs(responses) + 1e-12)
    aggregated = np.mean(normalized, axis=0)
    return estimate_distance_by_phase_slope(aggregated, freqs, round_trip=round_trip)["distance_est"]


def run(config: dict, overwrite: bool = False) -> dict:
    freqs, output_dir = setup_experiment(config, overwrite=overwrite)
    trial_rows = []
    summary_rows = []
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
            single_est = estimate_distance_by_phase_slope(responses[0], freqs, round_trip=config["round_trip"])["distance_est"]
            aggregate_est = _estimate_from_repeats(responses, freqs, config["round_trip"])
            return {
                "trial_id": seed,
                "true_distance": config["true_distance"],
                "single_est_distance": single_est,
                "est_distance": aggregate_est,
                "abs_error": abs(aggregate_est - config["true_distance"]),
                "single_abs_error": abs(single_est - config["true_distance"]),
                "n_repeats": n_repeats,
                "seed": seed,
            }
        results = run_monte_carlo(trial_fn, config["monte_carlo_trials"], config["random_seed"] + n_repeats * 17)
        df = collect_trial_results(results)
        trial_rows.append(df)
        summary = summarize_errors(df["true_distance"], df["est_distance"], thresholds=[config["success_threshold_m"]])
        summary_rows.append({"n_repeats": n_repeats, **summary, "single_rmse": float(np.sqrt(np.mean(df["single_abs_error"] ** 2)))})
    trial_df = pd.concat(trial_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(trial_df, output_dir / "tables", "trial_results.csv")
    save_dataframe(summary_df, output_dir / "tables", "summary_by_repeats.csv")
    plot_repeats_vs_error(summary_df["n_repeats"].to_numpy(), summary_df["rmse"].to_numpy(), output_dir / "figures" / "repeats_vs_rmse.png")
    plot_histogram_errors(trial_df["abs_error"].to_numpy(), output_dir / "figures" / "aggregate_error_histogram.png")
    grouped_errors = [trial_df[trial_df["n_repeats"] == repeat]["abs_error"].to_numpy() for repeat in config["n_repeats_list"]]
    plot_boxplot_errors(grouped_errors, [str(val) for val in config["n_repeats_list"]], output_dir / "figures" / "error_boxplot.png")
    return finish_experiment({"n_trials": int(len(trial_df)), "best_rmse": float(summary_df["rmse"].min())}, output_dir)
