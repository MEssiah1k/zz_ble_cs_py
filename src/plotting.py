"""Centralized plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_wrapped_unwrapped(freqs: np.ndarray, wrapped: np.ndarray, unwrapped: np.ndarray, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, wrapped, label="Wrapped phase")
    ax.plot(freqs, unwrapped, label="Unwrapped phase")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (rad)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_phase_fit(freqs: np.ndarray, unwrapped: np.ndarray, fitted: np.ndarray, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, unwrapped, "o", ms=3, label="Unwrapped phase")
    ax.plot(freqs, fitted, label="Linear fit")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (rad)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_true_vs_estimated(true_values: np.ndarray, est_values: np.ndarray, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_values, est_values, alpha=0.8)
    limits = [min(true_values.min(), est_values.min()), max(true_values.max(), est_values.max())]
    ax.plot(limits, limits, "--", color="black")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Estimated distance (m)")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_snr_vs_rmse(snr_values: np.ndarray, rmse_values: np.ndarray, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(snr_values, rmse_values, marker="o")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSE (m)")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_histogram_errors(errors: np.ndarray, output_path: str | Path, bins: int = 30) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=bins, alpha=0.8)
    ax.set_xlabel("Absolute error (m)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_boxplot_errors(groups: list[np.ndarray], labels: list[str], output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(groups, tick_labels=labels)
    ax.set_ylabel("Absolute error (m)")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_repeats_vs_error(repeats: np.ndarray, errors: np.ndarray, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(repeats, errors, marker="o")
    ax.set_xlabel("Repeats")
    ax.set_ylabel("RMSE (m)")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_num_devices_vs_error(num_devices: np.ndarray, errors: np.ndarray, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(num_devices, errors, marker="o")
    ax.set_xlabel("Concurrent devices")
    ax.set_ylabel("RMSE (m)")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_distance_gap_vs_success(distance_gaps: np.ndarray, success_rates: np.ndarray, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(distance_gaps, success_rates, marker="o")
    ax.set_xlabel("Distance gap (m)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_power_gap_vs_error(power_gaps: np.ndarray, errors: np.ndarray, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(power_gaps, errors, marker="o")
    ax.set_xlabel("Power gap (dB)")
    ax.set_ylabel("RMSE (m)")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_multipath_vs_error(path_values: np.ndarray, errors: np.ndarray, output_path: str | Path, xlabel: str = "Multipath parameter") -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(path_values, errors, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE (m)")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_kfactor_vs_error(k_values: np.ndarray, errors: np.ndarray, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, errors, marker="o")
    ax.set_xlabel("Rician K factor")
    ax.set_ylabel("RMSE (m)")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_score_curve(distance_grid: np.ndarray, scores: np.ndarray, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(distance_grid, scores)
    ax.set_xlabel("Hypothesized distance (m)")
    ax.set_ylabel("Coherent score")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(np.asarray(values, dtype=float), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_distance_gap_vs_error(
    distance_gaps: np.ndarray,
    errors: np.ndarray,
    output_path: str | Path,
    smooth_window: int = 5,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(distance_gaps, errors, marker="o", alpha=0.7, label="RMSE")
    if smooth_window > 1 and len(errors) >= smooth_window:
        smooth_errors = _moving_average(errors, smooth_window)
        ax.plot(distance_gaps, smooth_errors, linewidth=2.2, color="tab:red", label=f"Moving average ({smooth_window})")
    ax.set_xlabel("Device spacing (m)")
    ax.set_ylabel("RMSE (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, output_path)


def plot_error_heatmap(
    x_values: np.ndarray,
    y_values: np.ndarray,
    error_grid: np.ndarray,
    output_path: str | Path,
    xlabel: str,
    ylabel: str,
    colorbar_label: str = "RMSE (m)",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.imshow(
        np.asarray(error_grid, dtype=float),
        origin="lower",
        aspect="auto",
        extent=[float(np.min(x_values)), float(np.max(x_values)), float(np.min(y_values)), float(np.max(y_values))],
        interpolation="nearest",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)
    return _save(fig, output_path)
