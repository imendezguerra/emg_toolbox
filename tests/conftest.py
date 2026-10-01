"""Shared fixtures: small, deterministic synthetic EMG data."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

FS = 2048
DURATION_S = 2
N_SAMPLES = FS * DURATION_S
GRID = (4, 4)


def tone(freq_hz, amp=1.0, n_samples=N_SAMPLES, fs=FS):
    """Sine wave of shape (n_samples,)."""
    return amp * np.sin(2 * np.pi * freq_hz * np.arange(n_samples) / fs)


@pytest.fixture
def make_tone():
    """Factory for sine waves (see ``tone``)."""
    return tone


@pytest.fixture
def fs():
    return FS


@pytest.fixture
def timestamps():
    return np.arange(N_SAMPLES) / FS


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def ch_map():
    """(rows, cols) channel map with channels 0..15 in row-major order."""
    return np.arange(np.prod(GRID)).reshape(GRID)


@pytest.fixture
def emg(rng, ch_map):
    """(n_samples, chs) white noise, one column per channel in ``ch_map``."""
    return rng.standard_normal((N_SAMPLES, ch_map.size))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
