"""Functions to plot EMG signals"""

from copy import copy
from typing import Optional, Union
import numpy as np
from scipy import fft, signal
import seaborn as sns
import matplotlib.pyplot as plt

def plot_ch(
    data: np.ndarray,
    timestamps: np.ndarray,
    delta: Optional[int] = 5,
    ax: Optional[plt.Axes] = None,
    palette_name: Optional[str] = "viridis",
    **kwarg: Optional[Union[str, int, float]]
    )-> plt.Axes:
    
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


def plot_psd(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    
    """Plot the Power Spectral Density (PSD) of the input data.

    Args:
        data (np.ndarray): Input data with shape (samples, channels).
        fs (int, optional): Sampling frequency. Default is 2048.
        ax (plt.Axes, optional): Axes to plot the PSD. Default is None.

    Returns:
        plt.Axes: Axes where the PSD is plotted.
    """

    # Initialise variables
    samples = data.shape[0]

    # Compute PSD
    yf = fft.fft(data, axis=0)
    xf = fft.fftfreq(samples, 1/fs)[:samples//2]

    # Plot PSD
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10,5), layout='tight')
    ax.plot(xf, 2/samples * np.abs(yf[0:samples//2]))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    
    return ax

def plot_spectrogram(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    ax: Optional[plt.Axes] = None,        
) -> plt.Axes:
    
    """Plot the spectrogram of the input data.

    Args:
        data (np.ndarray): Input data with shape (samples, channels).
        fs (int, optional): Sampling frequency. Default is 2048.
        ax (plt.Axes, optional): Axes to plot the spectrogram. Default is None.

    Returns:
        plt.Axes: Axes where the spectrogram is plotted.
    """

    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(data, fs)

    # Plot spectrogram
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10,5), layout='tight')
    ax.pcolormesh(t, f, Sxx, shading='gouraud')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    
    return ax