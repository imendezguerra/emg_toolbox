"""Tools for EMG data processing"""

from copy import copy
import itertools
from typing import Union
import numpy as np

def replace_bad_ch(
    data: np.ndarray,
    bad_ch: Union[list, np.ndarray],
    ch_map: np.ndarray
    ) -> np.ndarray:
    
    """Replace bad channels by the mean of the neighbours

    Args:
        data (np.ndarray): Data to be processed with shape (samples, channels) or 
            (ch_rows, ch_cols, samples) 
        bad_ch (list, np.ndarray): Bad channels to be substituted with shape 
            (bad_channels,)
        ch_map (np.ndarray): Map of the channels in the electrode array with shape
            (ch_rows, ch_cols)

    Returns:
        out (np.ndarray): with shape (samples, channels) or (ch_rows, ch_cols, samples)
            Data with substituted bad channels
    """

    # Initialise variables
    rows, cols = ch_map.shape
    out = copy(data)

    # Substitute bad channels until there are no more bad channels left
    while len(bad_ch):

        # Get the current bad channel
        curr_bad_ch = bad_ch[0]

        # Find coordinated of the current bad channel
        idx = np.nonzero(curr_bad_ch == ch_map)
        x = idx[0][0]
        y = idx[1][0]

        # CASE A: [samples, ch] format
        if len(data.shape) == 2:

            # Select adjacent channels within range and not in the bad channel list
            adj_ch = []
            for row, col in itertools.product(range(x-1,x+2), range(y-1,y+2)):
                if row < 0 or row > rows:
                    continue
                if col < 0 or col > cols:
                    continue
                if ch_map[row, col] in bad_ch:
                    continue
                adj_ch.append( ch_map[row, col] )

            # Collect neighbouring channels
            adj_data = out[:, adj_ch]

        # CASE B: [ch_rows, ch_cols, samples] format
        elif len(data.shape) == 3:

            # Select adjacent channels within range and not in the bad channel list
            adj_ch_rc = []
            for row, col in itertools.product(range(x-1,x+2), range(y-1,y+2)):
                if row < 0 or row > rows:
                    continue
                if col < 0 or col > cols:
                    continue
                if ch_map[row, col] in bad_ch:
                    continue
                adj_ch_rc.append( (row, col) )

            # Collect neighbouring channels
            adj_data = np.empty(out.shape[-1], len(adj_ch_rc))
            for ch_rc, i in enumerate(adj_ch_rc):
                adj_data[:, i] = out[ch_rc]

        if np.size(adj_data):
            # Replace bad channel by the mean of the neighbours
            out[:, curr_bad_ch] = np.mean( adj_data, axis=-1 )

            # Remove current channel from the list
            bad_ch = np.delete(bad_ch, 0)
        else:
            # Send current bad channel to the end of the list
            bad_ch = np.roll(bad_ch, -1)

    return out
        