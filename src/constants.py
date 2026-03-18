"""Project constants."""

from __future__ import annotations

SPEED_OF_LIGHT = 3e8
DEFAULT_F_START_HZ = 2.402e9
DEFAULT_F_STEP_HZ = 1.0e6
DEFAULT_N_FREQS = 32
DEFAULT_RANDOM_SEED = 20260318
DEFAULT_MONTE_CARLO_TRIALS = 200
DEFAULT_DISTANCE_GRID_M = [round(x, 2) for x in [i * 0.05 for i in range(1, 401)]]
