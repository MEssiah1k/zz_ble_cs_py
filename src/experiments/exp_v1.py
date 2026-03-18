"""V1 single-trip vs round-trip phase-ranging comparison."""

from __future__ import annotations

import pandas as pd

from ..estimators import estimate_distance_by_phase_slope
from ..io_utils import save_dataframe
from ..plotting import plot_power_gap_vs_error, plot_true_vs_estimated
from ..signal_models import single_link_frequency_response
from .common import finish_experiment, setup_experiment


def run(config: dict, overwrite: bool = False) -> dict:
    freqs, output_dir = setup_experiment(config, overwrite=overwrite)
    rows = []
    for distance in config["distance_scan_values"]:
        for model_name, round_trip in [("single_trip", False), ("round_trip", True)]:
            for phase_offset in config["phase_offsets"]:
                response = single_link_frequency_response(freqs, distance, phase_offset=phase_offset, round_trip=round_trip)
                estimate = estimate_distance_by_phase_slope(response, freqs, round_trip=round_trip)
                rows.append(
                    {
                        "model": model_name,
                        "true_distance": distance,
                        "phase_offset": phase_offset,
                        "est_distance": estimate["distance_est"],
                        "abs_error": abs(estimate["distance_est"] - distance),
                    }
                )
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir / "tables", "single_vs_round_trip.csv")
    round_trip_df = df[df["model"] == "round_trip"]
    plot_true_vs_estimated(round_trip_df["true_distance"].to_numpy(), round_trip_df["est_distance"].to_numpy(), output_dir / "figures" / "round_trip_true_vs_estimated.png")
    phase_group = df.groupby("phase_offset", as_index=False)["abs_error"].mean()
    plot_power_gap_vs_error(phase_group["phase_offset"].to_numpy(), phase_group["abs_error"].to_numpy(), output_dir / "figures" / "phase_offset_vs_error.png")
    return finish_experiment({"rows": len(df), "mean_abs_error": float(df["abs_error"].mean())}, output_dir)
