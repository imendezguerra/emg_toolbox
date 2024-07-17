"""Tools for EMG data processing"""

from copy import copy
import itertools
from typing import Union, Tuple, Optional
import numpy as np
from scipy.interpolate import Akima1DInterpolator


def arrange_data_spatially(
    data: np.ndarray,
    ch_map: np.ndarray
    ) -> np.ndarray:

    """
    Arrange the data spatially based on the channel map.

    Args:
        data (np.ndarray): The input data array of shape (samples, chs).
        ch_map (np.ndarray): The channel map array of shape (rows, cols).

    Returns:
        np.ndarray: The spatially arranged data array of shape (rows, cols, samples).
    """

    rows, cols = ch_map.shape
    samples, chs = data.shape
    data_spatial = np.empty((rows, cols, samples))

    for row, col in itertools.product(range(rows), range(cols)):
        ch = ch_map[row, col]
        if (ch < 0) or (ch >= chs):
            data_spatial[row, col] = np.zeros(samples)
        else:
            data_spatial[row, col] = data[:, ch]

    return data_spatial


def flatten_data_spatially(
    data: np.ndarray,
    ch_map: np.ndarray
    ) -> np.ndarray:

    """
    Flatten the data spatially based on the channel map.

    Args:
        data (np.ndarray): The input data array of shape (rows, cols, samples).
        ch_map (np.ndarray): The channel map array of shape (rows, cols).

    Returns:
        np.ndarray: The spatially flattened data array of shape (samples, chs).
    """

    samples = data.shape[-1]
    chs = np.amax(ch_map) + 1
    
    data_flat = np.empty((samples, chs))

    for ch in range(chs):
        idx = np.where(ch_map == ch)
        if len(idx[0]):
            data_flat[:, ch] = data[idx[0], idx[1]]

    return data_flat


def replace_bad_ch(
    data: np.ndarray,
    bad_ch: Union[list, np.ndarray],
    ch_map: np.ndarray,
    ) -> np.ndarray:

    """
    Replace bad channels with the mean of their neighboring channels.

    Args:
        data (np.ndarray): Data with shape (samples, channels) or 
            (ch_rows, ch_cols, samples).
        bad_ch (Union[list, np.ndarray]): A list or numpy array containing the 
            indices of the bad channels.
        ch_map (np.ndarray): A 2D numpy array representing the channel map.

    Returns:
        data_out (np.ndarray): Data with the bad channels replaced.

    """

    # Initialise variables
    rows, cols = ch_map.shape
    data_out = copy(data)
    if isinstance(bad_ch, list):
        bad_ch_out = copy(bad_ch)
    else:
        bad_ch_out = copy(bad_ch).tolist()

    # Substitute bad channels until there are no more bad channels left
    while np.any(bad_ch_out):

        # Set updated bad channels to -1
        ch_map_out = copy(ch_map)
        for ch in bad_ch_out:
            ch_map_out[ ch_map == ch ] = -1

        # Get first bad channel
        curr_bad_ch = bad_ch_out.pop(0)

        # Find coordinates of the current bad channel
        [x, y] = np.where(ch_map == curr_bad_ch)
        x = x[0]
        y = y[0]
        
        # # Define mask
        neigh_mask = np.array([
            [x, y + 1],
            [x, y - 1],
            [x + 1, y],
            [x - 1, y],
            [x + 1, y + 1],
            [x + 1, y - 1],
            [x - 1, y + 1],
            [x - 1, y - 1]
            ])

        # Remove neighbours out of bounds
        in_range = np.logical_and(
            np.logical_and( neigh_mask[:,0] > -1, neigh_mask[:,0] < rows),
            np.logical_and( neigh_mask[:,1] > -1, neigh_mask[:,1] < cols)
        )
        neigh_mask = neigh_mask[in_range]
        in_range = in_range[in_range]

        # Check for good neighbours
        for i in range(neigh_mask.shape[0]):
            if (ch_map_out[ neigh_mask[i,0], neigh_mask[i,1] ] < 0):
                in_range[i] = False

        # Get the actual neighbouring channels
        neigh_mask = neigh_mask[in_range, :]
        neigh_chs = ch_map[neigh_mask[:,0], neigh_mask[:,1]]

        # Substitue channels
        if len(data_out.shape) == 2: # (samples, chs) format
            data_out[:, curr_bad_ch] = np.mean( data_out[:, neigh_chs], axis=-1)
        elif len(data_out.shape) == 3: # (ch_rows, ch_cols, samples) format
            data_out[x,y] = np.mean( data_out[neigh_mask[:,0], neigh_mask[:,1]], axis=[1,2])

    return data_out


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
        Tuple[np.ndarray, np.ndarray]: The RMS array and the corresponding timestamps.
    """

    # Initialise variables
    win_len = np.round( win_len_s * fs ).astype(int)
    win_step = np.round( win_step_s * fs ).astype(int)

    samples, chs = data.shape
    win_num = np.round( (samples - win_len)/win_step ).astype(int) + 1

    timestamps_aux = np.linspace( timestamps[0], timestamps[-1], win_num)
    rms_aux = np.zeros((win_num, chs))

    # Compute RMS
    for win in range(win_num):
        mask = np.arange(
            0 + win_step * win,
            np.amin([win_len + win_step * win, samples])
            ).astype(int)
        rms_aux[win] = np.sqrt(np.mean(data[mask]**2, axis=0))

    # Interpolate RMS to match data signals
    rms = np.zeros_like(data)
    for ch in range(chs):
        rms[:, ch] = Akima1DInterpolator(timestamps_aux, rms_aux[:, ch])(timestamps)

    return rms, timestamps_aux
