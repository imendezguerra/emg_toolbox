"""Functions to compute features from EMG signals"""

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import Akima1DInterpolator


def compute_rms(
    data: np.ndarray,
    timestamps: np.ndarray,
    win_len_s: Optional[float] = 0.1,
    win_step_s: Optional[float] = 0.1,
    fs: Optional[int] = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Compute the Root Mean Square (RMS) of the data.

    Args:
        data (np.ndarray): The input data array of shape (samples, chs).
        timestamps (np.ndarray): The timestamps array.
        win_len_s (float, optional): The window length in seconds. Defaults to 0.1.
        win_step_s (float, optional): The window step in seconds. Defaults to 0.1.
        fs (int, optional): The sampling frequency. Defaults to 2048.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The RMS array interpolated to the input
            timestamps with shape (samples, chs), and the timestamps of the
            window centres with shape (win_num,).
    """

    # Initialise variables
    samples, chs = data.shape
    win_len = min(np.round( win_len_s * fs ).astype(int), samples)
    win_step = np.round( win_step_s * fs ).astype(int)

    win_num = np.round( (samples - win_len)/win_step ).astype(int) + 1

    timestamps_aux = np.zeros(win_num)
    rms_aux = np.zeros((win_num, chs))

    # Compute RMS
    for win in range(win_num):
        mask = np.arange(
            0 + win_step * win,
            np.amin([win_len + win_step * win, samples])
            ).astype(int)
        rms_aux[win] = np.sqrt(np.mean(data[mask]**2, axis=0))
        timestamps_aux[win] = np.mean(timestamps[mask])

    # Single window, no interpolation possible
    if win_num < 2:
        rms = np.tile(rms_aux, (samples, 1))
        return rms, timestamps_aux

    # Interpolate RMS to match data signals, holding the edge values before
    # the first and after the last window centre
    timestamps_interp = np.clip(timestamps, timestamps_aux[0], timestamps_aux[-1])
    rms = np.zeros_like(data, dtype=float)
    for ch in range(chs):
        rms[:, ch] = Akima1DInterpolator(timestamps_aux, rms_aux[:, ch])(timestamps_interp)

    return rms, timestamps_aux
