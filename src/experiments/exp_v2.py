"""V2 AWGN robustness experiment."""

from __future__ import annotations

import pandas as pd

from ..channel_models import add_awgn
from ..estimators import estimate_distance_by_phase_slope
from ..io_utils import save_dataframe
from ..metrics import summarize_errors
from ..monte_carlo import collect_trial_results, run_monte_carlo
from ..plotting import plot_boxplot_errors, plot_histogram_errors, plot_snr_vs_rmse
from ..signal_models import single_link_frequency_response
from ..utils import set_random_seed
from .common import finish_experiment, setup_experiment


def run(config: dict, overwrite: bool = False) -> dict:
    freqs, output_dir = setup_experiment(config, overwrite=overwrite)
    trial_rows = []
    summary_rows = []
    for snr_db in config["snr_db_list"]:
        def trial_fn(seed: int) -> dict:
            rng = set_random_seed(seed)
            clean = single_link_frequency_response(freqs, config["true_distance"], round_trip=config["round_trip"])
            noisy = add_awgn(clean, snr_db=snr_db, rng=rng)
            estimate = estimate_distance_by_phase_slope(noisy, freqs, round_trip=config["round_trip"])
            return {
                "trial_id": seed,
                "true_distance": config["true_distance"],
                "est_distance": estimate["distance_est"],
                "abs_error": abs(estimate["distance_est"] - config["true_distance"]),
                "sq_error": (estimate["distance_est"] - config["true_distance"]) ** 2,
                "snr_db": snr_db,
                "n_freqs": config["n_freqs"],
                "f_step": config["f_step"],
                "seed": seed,
            }
        results = run_monte_carlo(trial_fn, config["monte_carlo_trials"], config["random_seed"] + int(10 * snr_db + 100))
        df = collect_trial_results(results)
        trial_rows.append(df)
        summary = summarize_errors(df["true_distance"], df["est_distance"], thresholds=[config["success_threshold_m"]])
        summary_rows.append({"snr_db": snr_db, **summary})
    trial_df = pd.concat(trial_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(trial_df, output_dir / "tables", "trial_results.csv")
    save_dataframe(summary_df, output_dir / "tables", "summary_by_snr.csv")
    plot_snr_vs_rmse(summary_df["snr_db"].to_numpy(), summary_df["rmse"].to_numpy(), output_dir / "figures" / "snr_vs_rmse.png")
    plot_histogram_errors(trial_df["abs_error"].to_numpy(), output_dir / "figures" / "error_histogram.png")
    grouped_errors = [trial_df[trial_df["snr_db"] == snr]["abs_error"].to_numpy() for snr in config["snr_db_list"]]
    plot_boxplot_errors(grouped_errors, [str(snr) for snr in config["snr_db_list"]], output_dir / "figures" / "error_boxplot.png")
    return finish_experiment({"n_trials": int(len(trial_df)), "best_rmse": float(summary_df["rmse"].min())}, output_dir)
