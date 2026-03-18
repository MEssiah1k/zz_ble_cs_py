"""Monte Carlo helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd


def run_monte_carlo(trial_fn: Callable[[int], dict[str, Any]], n_trials: int, seed: int) -> list[dict[str, Any]]:
    return [trial_fn(seed + trial_id) for trial_id in range(n_trials)]


def collect_trial_results(results_list: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(results_list)


def summarize_trial_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    summary = {"n_rows": int(len(df))}
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        summary[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std(ddof=0)),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
    return summary
