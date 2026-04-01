"""Distance estimation methods."""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import SPEED_OF_LIGHT
from .target_matching import estimate_target_distance_by_scan, estimate_target_distance_from_peer_pair
from .utils import unwrap_phase_safe


def estimate_distance_by_phase_slope(response: np.ndarray, freqs: np.ndarray, round_trip: bool = True) -> dict[str, Any]:
    response = np.asarray(response, dtype=np.complex128)
    if response.ndim != 1:
        raise ValueError("Phase slope estimator expects a single response vector")
    phase_wrapped = np.angle(response)
    phase_unwrapped = unwrap_phase_safe(phase_wrapped)
    slope, intercept = np.polyfit(freqs, phase_unwrapped, 1)
    tau = -slope / (2.0 * np.pi)
    divisor = 2.0 if round_trip else 1.0
    distance_est = SPEED_OF_LIGHT * tau / divisor
    fitted_phase = slope * freqs + intercept
    return {
        "distance_est": float(distance_est),
        "tau_est": float(tau),
        "slope": float(slope),
        "intercept": float(intercept),
        "phase_wrapped": phase_wrapped,
        "phase_unwrapped": phase_unwrapped,
        "phase_fit": fitted_phase,
    }


def estimate_distance_by_target_scan(
    response: np.ndarray,
    freqs: np.ndarray,
    distance_grid: np.ndarray,
    round_trip: bool = True,
    score_mode: str = "composite",
    prior_distance: float | None = None,
    prior_sigma_m: float | None = None,
) -> dict[str, Any]:
    return estimate_target_distance_by_scan(
        response,
        freqs,
        distance_grid,
        round_trip=round_trip,
        score_mode=score_mode,
        prior_distance=prior_distance,
        prior_sigma_m=prior_sigma_m,
    )


def estimate_distance_batch(responses: np.ndarray, freqs: np.ndarray, method: str, **kwargs: Any) -> list[dict[str, Any]]:
    responses = np.asarray(responses)
    estimators = {
        "phase_slope": estimate_distance_by_phase_slope,
        "target_scan": estimate_distance_by_target_scan,
    }
    if method not in estimators:
        raise ValueError(f"Unsupported method: {method}")
    estimator = estimators[method]
    return [estimator(response, freqs, **kwargs) for response in responses]


def estimate_distance_by_peer_multifreq(
    local_iq: np.ndarray,
    peer_iq: np.ndarray,
    freqs: np.ndarray,
    distance_grid: np.ndarray,
    round_trip: bool = True,
    score_mode: str = "composite",
) -> dict[str, Any]:
    return estimate_target_distance_from_peer_pair(
        local_iq,
        peer_iq,
        freqs,
        distance_grid,
        round_trip=round_trip,
        score_mode=score_mode,
    )
