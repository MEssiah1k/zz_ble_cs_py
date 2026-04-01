"""V6 indoor fading experiment."""

from __future__ import annotations

import pandas as pd

from ..channel_models import add_awgn, apply_rayleigh_fading, apply_rician_fading
from ..estimators import estimate_distance_by_target_scan
from ..io_utils import save_dataframe
from ..metrics import summarize_errors
from ..monte_carlo import collect_trial_results, run_monte_carlo
from ..plotting import plot_histogram_errors, plot_kfactor_vs_error
from ..signal_models import multi_link_frequency_response
from ..utils import set_random_seed
from .common import config_distance_grid, finish_experiment, setup_experiment


def run(config: dict, overwrite: bool = False) -> dict:
    freqs, output_dir = setup_experiment(config, overwrite=overwrite)
    distance_grid = config_distance_grid(config)
    k_rows = []
    all_trials = []

    distances = [config["base_target_distance"] + idx * 2.0 for idx in range(config["num_devices"])]
    amplitudes = [1.0] * config["num_devices"]

    for k_factor in config["rician_k_list"]:
        results = run_monte_carlo(
            lambda seed: _trial(
                seed,
                freqs,
                distance_grid,
                distances,
                amplitudes,
                config["base_target_distance"],
                config["snr_db"],
                k_factor,
                config["round_trip"],
                config.get("target_prior_sigma_m"),
            ),
            config["channel_realizations"],
            config["random_seed"] + int(k_factor * 59),
        )
        df = collect_trial_results(results)
        all_trials.append(df)
        metrics = summarize_errors(df["true_distance"], df["est_distance"], thresholds=[config["success_threshold_m"]])
        k_rows.append({"k_factor": k_factor, **metrics})

    rayleigh_results = run_monte_carlo(
        lambda seed: _trial(
            seed,
            freqs,
            distance_grid,
            distances,
            amplitudes,
            config["base_target_distance"],
            config["snr_db"],
            None,
            config["round_trip"],
            config.get("target_prior_sigma_m"),
        ),
        config["channel_realizations"],
        config["random_seed"] + 9991,
    )
    rayleigh_df = collect_trial_results(rayleigh_results)
    all_trials.append(rayleigh_df.assign(channel="rayleigh"))
    k_df = pd.DataFrame(k_rows)
    trial_df = pd.concat(all_trials, ignore_index=True)
    save_dataframe(k_df, output_dir / "tables", "summary_k_factor.csv")
    save_dataframe(trial_df, output_dir / "tables", "trial_results.csv")
    plot_kfactor_vs_error(k_df["k_factor"].to_numpy(), k_df["rmse"].to_numpy(), output_dir / "figures" / "kfactor_vs_rmse.png")
    plot_histogram_errors(trial_df["abs_error"].to_numpy(), output_dir / "figures" / "error_histogram.png")
    return finish_experiment({"best_rmse": float(k_df["rmse"].min())}, output_dir)


def _trial(seed: int, freqs, distance_grid, distances, amplitudes, true_distance, snr_db, k_factor, round_trip, target_prior_sigma_m):
    rng = set_random_seed(seed)
    response = multi_link_frequency_response(freqs, distances, amplitudes, phase_offsets=rng.uniform(0.0, 2.0 * 3.141592653589793, size=len(distances)), round_trip=round_trip)
    if k_factor is None:
        faded = apply_rayleigh_fading(response, rng=rng)
        channel = "rayleigh"
    else:
        faded = apply_rician_fading(response, k_factor=k_factor, rng=rng)
        channel = "rician"
    noisy = add_awgn(faded, snr_db=snr_db, rng=rng)
    estimate = estimate_distance_by_target_scan(
        noisy,
        freqs,
        distance_grid,
        round_trip=round_trip,
        prior_distance=true_distance,
        prior_sigma_m=target_prior_sigma_m,
    )
    return {
        "trial_id": seed,
        "true_distance": true_distance,
        "est_distance": estimate["distance_est"],
        "abs_error": abs(estimate["distance_est"] - true_distance),
        "k_factor": -1 if k_factor is None else k_factor,
        "channel": channel,
        "seed": seed,
    }
