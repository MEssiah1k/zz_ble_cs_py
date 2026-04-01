"""Signal generation for single-link and multi-link PBR-style responses."""

from __future__ import annotations

import numpy as np

from .constants import SPEED_OF_LIGHT
from .impairments import random_phase_offsets


def _propagation_delay(distance: float, round_trip: bool = True) -> float:
    factor = 2.0 if round_trip else 1.0
    return factor * float(distance) / SPEED_OF_LIGHT


def single_link_frequency_response(
    freqs: np.ndarray,
    distance: float,
    amplitude: float = 1.0,
    phase_offset: float | np.ndarray = 0.0,
    round_trip: bool = True,
) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    tau = _propagation_delay(distance, round_trip=round_trip)
    phase = -2.0 * np.pi * freqs * tau + np.asarray(phase_offset)
    return amplitude * np.exp(1j * phase)


def multi_link_frequency_response(
    freqs: np.ndarray,
    distances: np.ndarray,
    amplitudes: np.ndarray,
    phase_offsets: np.ndarray | None = None,
    round_trip: bool = True,
) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    distances = np.asarray(distances, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    phase_offsets = np.zeros(distances.shape[0], dtype=float) if phase_offsets is None else np.asarray(phase_offsets, dtype=float)
    response = np.zeros(freqs.shape[0], dtype=np.complex128)
    for distance, amplitude, phase_offset in zip(distances, amplitudes, phase_offsets):
        response += single_link_frequency_response(freqs, distance, amplitude, phase_offset, round_trip=round_trip)
    return response


def repeated_measurements_single_link(
    freqs: np.ndarray,
    distance: float,
    n_repeats: int,
    amplitude: float = 1.0,
    random_phase: bool = False,
    round_trip: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    responses = []
    # V3 关注的是“每次测量一个随机相位偏置”，而不是每个频点独立乱跳。
    phase_offsets = random_phase_offsets(n_repeats, 1, len(freqs), rng, per_frequency=False).reshape(n_repeats, len(freqs)) if random_phase else np.zeros((n_repeats, len(freqs)))
    for repeat_idx in range(n_repeats):
        responses.append(
            single_link_frequency_response(
                freqs,
                distance,
                amplitude=amplitude,
                phase_offset=phase_offsets[repeat_idx],
                round_trip=round_trip,
            )
        )
    return np.asarray(responses)


def repeated_measurements_multi_link(
    freqs: np.ndarray,
    distances: np.ndarray,
    amplitudes: np.ndarray,
    n_repeats: int,
    random_phase: bool = True,
    round_trip: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    distances = np.asarray(distances, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    phase_offsets = random_phase_offsets(n_repeats, len(distances), len(freqs), rng) if random_phase else np.zeros((n_repeats, len(distances), len(freqs)))
    responses = []
    for repeat_idx in range(n_repeats):
        response = np.zeros(len(freqs), dtype=np.complex128)
        for dev_idx, (distance, amplitude) in enumerate(zip(distances, amplitudes)):
            response += single_link_frequency_response(
                freqs,
                distance,
                amplitude=amplitude,
                phase_offset=phase_offsets[repeat_idx, dev_idx],
                round_trip=round_trip,
            )
        responses.append(response)
    return np.asarray(responses)


def build_paired_iq_multi_link(
    freqs: np.ndarray,
    distances: np.ndarray,
    amplitudes: np.ndarray,
    phase_offsets: np.ndarray | None = None,
    round_trip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(freqs, dtype=float)
    distances = np.asarray(distances, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if phase_offsets is None:
        phase_offsets_arr = np.zeros((len(distances), len(freqs)), dtype=float)
    else:
        phase_offsets_arr = np.asarray(phase_offsets, dtype=float)
        if phase_offsets_arr.ndim == 1:
            phase_offsets_arr = np.repeat(phase_offsets_arr[:, None], len(freqs), axis=1)
    local_iq = np.zeros(len(freqs), dtype=np.complex128)
    peer_iqs = np.zeros((len(distances), len(freqs)), dtype=np.complex128)
    for dev_idx, (distance, amplitude) in enumerate(zip(distances, amplitudes)):
        offset = phase_offsets_arr[dev_idx]
        local_iq += single_link_frequency_response(freqs, distance, amplitude=amplitude, phase_offset=offset, round_trip=round_trip)
        peer_iqs[dev_idx] = np.exp(1j * offset)
    return local_iq, peer_iqs
