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


def _projection_score(compensated: np.ndarray) -> float:
    if compensated.ndim == 1:
        coherent_power = np.abs(np.mean(compensated)) ** 2
        total_power = np.mean(np.abs(compensated) ** 2) + 1e-12
        return float(coherent_power / total_power)
    coherent_power = np.abs(np.mean(compensated, axis=-1)) ** 2
    total_power = np.mean(np.abs(compensated) ** 2, axis=-1) + 1e-12
    return float(np.mean(coherent_power / total_power))


def _adjacent_phase_consistency(compensated: np.ndarray) -> float:
    if compensated.shape[-1] < 2:
        return 0.0
    deltas = compensated[..., 1:] * np.conj(compensated[..., :-1])
    normalized = deltas / (np.abs(deltas) + 1e-12)
    if normalized.ndim == 1:
        return float(np.abs(np.mean(normalized)))
    return float(np.mean(np.abs(np.mean(normalized, axis=-1))))


def _combined_score(compensated: np.ndarray) -> tuple[float, float, float]:
    projection_score = _projection_score(compensated)
    adjacent_score = _adjacent_phase_consistency(compensated)
    combined = 0.8 * projection_score + 0.2 * adjacent_score
    return combined, projection_score, adjacent_score


def _peak_diagnostics(distance_grid: np.ndarray, scores: np.ndarray, best_idx: int) -> dict[str, float]:
    if len(scores) <= 1:
        return {
            "best_score": float(scores[best_idx]),
            "second_best_score": 0.0,
            "peak_margin": float(scores[best_idx]),
            "peak_ratio": float(scores[best_idx] / 1e-12),
            "confidence": 1.0,
        }
    step = float(distance_grid[1] - distance_grid[0]) if len(distance_grid) > 1 else 0.05
    exclude_radius = max(0.25, 2.0 * step)
    neighbor_mask = np.abs(distance_grid - distance_grid[best_idx]) <= exclude_radius
    candidate_scores = scores[~neighbor_mask]
    second_best = float(np.max(candidate_scores)) if candidate_scores.size else 0.0
    best_score = float(scores[best_idx])
    peak_margin = best_score - second_best
    peak_ratio = best_score / (second_best + 1e-12)
    confidence = peak_margin / (best_score + 1e-12)
    return {
        "best_score": best_score,
        "second_best_score": second_best,
        "peak_margin": float(peak_margin),
        "peak_ratio": float(peak_ratio),
        "confidence": float(confidence),
    }


def _build_prior_weights(distance_grid: np.ndarray, prior_distance: float | None, prior_sigma_m: float | None) -> np.ndarray:
    if prior_distance is None or prior_sigma_m is None or prior_sigma_m <= 0.0:
        return np.ones_like(distance_grid, dtype=float)
    normalized = (distance_grid - float(prior_distance)) / float(prior_sigma_m)
    return np.exp(-0.5 * normalized * normalized)


def scan_distance_hypotheses(
    response: np.ndarray,
    freqs: np.ndarray,
    distance_grid: np.ndarray,
    round_trip: bool = True,
    score_mode: str = "composite",
    prior_distance: float | None = None,
    prior_sigma_m: float | None = None,
) -> dict[str, np.ndarray]:
    distance_grid = np.asarray(distance_grid, dtype=float)
    legacy_scores = []
    projection_scores = []
    adjacent_scores = []
    combined_scores = []
    for candidate in distance_grid:
        compensated = apply_target_compensation(response, freqs, candidate, round_trip=round_trip)
        legacy_scores.append(coherent_score_after_compensation(response, freqs, candidate, round_trip=round_trip))
        combined, projection_score, adjacent_score = _combined_score(compensated)
        combined_scores.append(combined)
        projection_scores.append(projection_score)
        adjacent_scores.append(adjacent_score)
    score_map = {
        "legacy": np.asarray(legacy_scores, dtype=float),
        "projection": np.asarray(projection_scores, dtype=float),
        "adjacent": np.asarray(adjacent_scores, dtype=float),
        "composite": np.asarray(combined_scores, dtype=float),
    }
    if score_mode not in score_map:
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    prior_weights = _build_prior_weights(distance_grid, prior_distance, prior_sigma_m)
    return {
        "distance_grid": distance_grid,
        "scores": score_map[score_mode] * prior_weights,
        "raw_scores": score_map[score_mode],
        "legacy_scores": score_map["legacy"],
        "projection_scores": score_map["projection"],
        "adjacent_scores": score_map["adjacent"],
        "composite_scores": score_map["composite"],
        "prior_weights": prior_weights,
    }


def estimate_target_distance_by_scan(
    response: np.ndarray,
    freqs: np.ndarray,
    distance_grid: np.ndarray,
    round_trip: bool = True,
    score_mode: str = "composite",
    prior_distance: float | None = None,
    prior_sigma_m: float | None = None,
) -> dict[str, np.ndarray | float]:
    scan = scan_distance_hypotheses(
        response,
        freqs,
        distance_grid,
        round_trip=round_trip,
        score_mode=score_mode,
        prior_distance=prior_distance,
        prior_sigma_m=prior_sigma_m,
    )
    best_idx = int(np.argmax(scan["scores"]))
    diagnostics = _peak_diagnostics(scan["distance_grid"], scan["scores"], best_idx)
    result = {
        "distance_est": float(scan["distance_grid"][best_idx]),
        "distance_grid": scan["distance_grid"],
        "scores": scan["scores"],
        "raw_scores": scan["raw_scores"],
        "score_mode": score_mode,
        "legacy_scores": scan["legacy_scores"],
        "projection_scores": scan["projection_scores"],
        "adjacent_scores": scan["adjacent_scores"],
        "composite_scores": scan["composite_scores"],
        "prior_weights": scan["prior_weights"],
    }
    result.update(diagnostics)
    return result
