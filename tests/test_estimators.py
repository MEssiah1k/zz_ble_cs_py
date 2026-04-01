import numpy as np

from src.estimators import estimate_distance_by_peer_multifreq, estimate_distance_by_phase_slope
from src.signal_models import build_paired_iq_multi_link, single_link_frequency_response


def test_phase_slope_estimator_recovers_distance_without_noise():
    freqs = np.linspace(2.402e9, 2.45e9, 64)
    true_distance = 7.5
    response = single_link_frequency_response(freqs, distance=true_distance, round_trip=True)
    estimate = estimate_distance_by_phase_slope(response, freqs, round_trip=True)
    assert abs(estimate["distance_est"] - true_distance) < 1e-6


def test_peer_multifrequency_estimator_recovers_target_without_noise():
    freqs = np.linspace(2.402e9, 2.45e9, 64)
    local_iq, peer_iqs = build_paired_iq_multi_link(
        freqs,
        [7.5, 12.0],
        [1.0, 0.25],
        phase_offsets=np.vstack(
            [
                np.linspace(0.2, 1.0, len(freqs)),
                np.linspace(1.4, 2.2, len(freqs)),
            ]
        ),
        round_trip=True,
    )
    estimate = estimate_distance_by_peer_multifreq(local_iq, peer_iqs[0], freqs, np.arange(1.0, 15.0, 0.05), round_trip=True)
    assert abs(estimate["distance_est"] - 7.5) < 0.25
