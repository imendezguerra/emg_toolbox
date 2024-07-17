"""Functions to plot EMG signals"""

from copy import copy
from typing import Optional, Union
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_ch(
    data: np.ndarray,
    timestamps: np.ndarray,
    delta: Optional[int] = 5,
    ax: Optional[plt.Axes] = None,
    palette_name: Optional[str] = "viridis",
    **kwarg: Optional[Union[str, int, float]]
    ):
    """Plot data per channel in a single plot

    Args:
        data (np.ndarray): Data to be displayed. It should have shape 
            (samples, channels).
        timestamps (np.ndarray): Data timestamps. It should have shape (samples,).
        delta (int, optional): Gain applied to the maximum of the data to modulate
            space between channels. Default is 5.
        ax (plt.Axes, optional): Axes to plot the data. Default is None.
        palette_name (str, optional): Name of the seaborn palette used for plotting.
            Default is "viridis".
        **kwarg: Additional keyword arguments to be passed to the `plot` function of
            matplotlib.

    Returns:
        ax (plt.Axes): Axes where the data is plotted.
    """

    # Data size
    chs = data.shape[1]

    # Compute channel offset
    max_val = np.nanmax(data)
    offset = max_val/delta

    # Apply offset to signals
    data_plot = copy(data)
    data_plot += range(0,chs) * offset

    # Plot signals
    if ax is None:
        _, ax = plt.subplots()

    # Create color palette and assign it to the axis cycle
    color_palette = sns.color_palette(palette_name, chs).as_hex()
    ax.set_prop_cycle(color=color_palette)

    ax.plot(timestamps, data_plot, **kwarg)
    ax.margins(0) # padding for each limit of the axis
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channels')
    ax.grid(True, axis='x')
    ch_ticks = np.linspace(0, (chs -1) * offset, chs)
    sel_chs = np.arange(0, chs, 5)
    ax.set_yticks(ch_ticks[sel_chs], sel_chs)

    return ax
