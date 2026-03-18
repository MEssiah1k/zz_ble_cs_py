"""I/O helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir, write_json


def prepare_output_dir(base_dir: str | Path, version_name: str, overwrite: bool = False) -> Path:
    root = ensure_dir(base_dir)
    output_dir = root / version_name
    if overwrite and output_dir.exists():
        for child in output_dir.glob("*"):
            if child.is_file():
                child.unlink()
    ensure_dir(output_dir)
    return output_dir


def save_config_snapshot(config: dict[str, Any], output_dir: str | Path, filename: str = "config_snapshot.json") -> Path:
    path = Path(output_dir) / filename
    write_json(path, config)
    return path


def save_dataframe(df: pd.DataFrame, output_dir: str | Path, filename: str) -> Path:
    path = Path(output_dir) / filename
    df.to_csv(path, index=False)
    return path


def save_summary(summary: dict[str, Any], output_dir: str | Path, filename: str = "summary.json") -> Path:
    path = Path(output_dir) / filename
    write_json(path, summary)
    return path
