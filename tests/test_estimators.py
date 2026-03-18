import numpy as np

from src.estimators import estimate_distance_by_phase_slope
from src.signal_models import single_link_frequency_response


def test_phase_slope_estimator_recovers_distance_without_noise():
    freqs = np.linspace(2.402e9, 2.45e9, 64)
    true_distance = 7.5
    response = single_link_frequency_response(freqs, distance=true_distance, round_trip=True)
    estimate = estimate_distance_by_phase_slope(response, freqs, round_trip=True)
    assert abs(estimate["distance_est"] - true_distance) < 1e-6
