"""V0 ideal single-link phase-ranging experiment."""

from __future__ import annotations

import pandas as pd

from ..estimators import estimate_distance_by_phase_slope
from ..frequency_plan import build_frequency_grid
from ..io_utils import save_dataframe
from ..metrics import absolute_error
from ..plotting import plot_phase_fit, plot_phase_wrapped_unwrapped, plot_true_vs_estimated
from ..signal_models import single_link_frequency_response
from .common import finish_experiment, setup_experiment


def run(config: dict, overwrite: bool = False) -> dict:
    freqs, output_dir = setup_experiment(config, overwrite=overwrite)
    rows = []
    for distance in config["distance_scan_values"]:
        response = single_link_frequency_response(freqs, distance, round_trip=config["round_trip"])
        estimate = estimate_distance_by_phase_slope(response, freqs, round_trip=config["round_trip"])
        rows.append(
            {
                "true_distance": distance,
                "est_distance": estimate["distance_est"],
                "abs_error": absolute_error(distance, estimate["distance_est"]),
                "n_freqs": config["n_freqs"],
                "f_step": config["f_step"],
            }
        )
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir / "tables", "distance_scan.csv")

    reference_distance = config["distance_scan_values"][len(config["distance_scan_values"]) // 2]
    reference_response = single_link_frequency_response(freqs, reference_distance, round_trip=config["round_trip"])
    reference_est = estimate_distance_by_phase_slope(reference_response, freqs, round_trip=config["round_trip"])
    plot_phase_wrapped_unwrapped(freqs, reference_est["phase_wrapped"], reference_est["phase_unwrapped"], output_dir / "figures" / "phase_wrapped_unwrapped.png")
    plot_phase_fit(freqs, reference_est["phase_unwrapped"], reference_est["phase_fit"], output_dir / "figures" / "phase_fit.png")
    plot_true_vs_estimated(df["true_distance"].to_numpy(), df["est_distance"].to_numpy(), output_dir / "figures" / "true_vs_estimated.png")

    cfg_rows = []
    for n_freqs in config["n_freqs_list"]:
        for f_step in config["f_step_list"]:
            local_freqs = build_frequency_grid(config["f_start"], f_step, n_freqs)
            response = single_link_frequency_response(local_freqs, reference_distance, round_trip=config["round_trip"])
            estimate = estimate_distance_by_phase_slope(response, local_freqs, round_trip=config["round_trip"])
            cfg_rows.append(
                {
                    "true_distance": reference_distance,
                    "est_distance": estimate["distance_est"],
                    "abs_error": absolute_error(reference_distance, estimate["distance_est"]),
                    "n_freqs": n_freqs,
                    "f_step": f_step,
                }
            )
    save_dataframe(pd.DataFrame(cfg_rows), output_dir / "tables", "frequency_config_scan.csv")
    return finish_experiment({"rows": len(df), "mean_abs_error": float(df["abs_error"].mean())}, output_dir)
