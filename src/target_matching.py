"""Target-link matching and scan-based detection."""

from __future__ import annotations

import numpy as np

from .signal_models import _propagation_delay


def build_target_compensation(freqs: np.ndarray, hypothesized_distance: float, round_trip: bool = True) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    tau = _propagation_delay(hypothesized_distance, round_trip=round_trip)
    return np.exp(1j * 2.0 * np.pi * freqs * tau)


def apply_target_compensation(response: np.ndarray, freqs: np.ndarray, hypothesized_distance: float, round_trip: bool = True) -> np.ndarray:
    compensation = build_target_compensation(freqs, hypothesized_distance, round_trip=round_trip)
    return np.asarray(response, dtype=np.complex128) * compensation


def coherent_score_after_compensation(response: np.ndarray, freqs: np.ndarray, hypothesized_distance: float, round_trip: bool = True) -> float:
    compensated = apply_target_compensation(response, freqs, hypothesized_distance, round_trip=round_trip)
    if compensated.ndim == 1:
        coherent_sum = np.abs(np.sum(compensated))
        energy = np.sum(np.abs(compensated)) + 1e-12
        return float(coherent_sum / energy)
    coherent_sum = np.abs(np.sum(compensated, axis=-1))
    energy = np.sum(np.abs(compensated), axis=-1) + 1e-12
    return float(np.mean(coherent_sum / energy))


def scan_distance_hypotheses(response: np.ndarray, freqs: np.ndarray, distance_grid: np.ndarray, round_trip: bool = True) -> dict[str, np.ndarray]:
    distance_grid = np.asarray(distance_grid, dtype=float)
    scores = np.asarray(
        [coherent_score_after_compensation(response, freqs, candidate, round_trip=round_trip) for candidate in distance_grid],
        dtype=float,
    )
    return {"distance_grid": distance_grid, "scores": scores}


def estimate_target_distance_by_scan(response: np.ndarray, freqs: np.ndarray, distance_grid: np.ndarray, round_trip: bool = True) -> dict[str, np.ndarray | float]:
    scan = scan_distance_hypotheses(response, freqs, distance_grid, round_trip=round_trip)
    best_idx = int(np.argmax(scan["scores"]))
    return {
        "distance_est": float(scan["distance_grid"][best_idx]),
        "best_score": float(scan["scores"][best_idx]),
        "distance_grid": scan["distance_grid"],
        "scores": scan["scores"],
    }
