import numpy as np

from src.signal_models import build_paired_iq_multi_link, multi_link_frequency_response, single_link_frequency_response
from src.target_matching import estimate_target_distance_by_scan


def test_single_link_frequency_response_has_unit_amplitude_without_impairment():
    freqs = np.linspace(2.402e9, 2.43e9, 32)
    response = single_link_frequency_response(freqs, distance=5.0, amplitude=1.0, round_trip=True)
    assert np.allclose(np.abs(response), 1.0, atol=1e-9)


def test_multi_device_target_peak_exists_when_target_is_stronger():
    freqs = np.linspace(2.402e9, 2.45e9, 48)
    response = multi_link_frequency_response(freqs, [4.0, 9.0], [1.0, 0.3], [0.0, 0.0], round_trip=True)
    distance_grid = np.arange(1.0, 12.0, 0.05)
    estimate = estimate_target_distance_by_scan(response, freqs, distance_grid, round_trip=True)
    assert abs(estimate["distance_est"] - 4.0) < 0.25


def test_paired_iq_model_recovers_target_after_peer_compensation():
    freqs = np.linspace(2.402e9, 2.45e9, 48)
    phase_offsets = np.vstack(
        [
            np.linspace(0.1, 1.1, len(freqs)),
            np.linspace(1.7, 2.7, len(freqs)),
        ]
    )
    local_iq, peer_iqs = build_paired_iq_multi_link(freqs, [4.0, 9.0], [1.0, 0.3], phase_offsets=phase_offsets, round_trip=True)
    matched = local_iq * np.conj(peer_iqs[0])
    estimate = estimate_target_distance_by_scan(matched, freqs, np.arange(1.0, 12.0, 0.05), round_trip=True)
    assert abs(estimate["distance_est"] - 4.0) < 0.25
