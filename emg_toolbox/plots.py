"""Functions to plot EMG signals"""

from copy import copy
from typing import Optional, Union
import numpy as np
from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from emg_toolbox.freq import get_psd


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
    palette_name: Optional[str] = "viridis",
    log_scale: Optional[bool] = False,
    **kwarg: Optional[Union[str, int, float]]
) -> plt.Axes:

    """Plot the Power Spectral Density (PSD) of the input data.

    Args:
        data (np.ndarray): Input data with shape (samples, channels).
        fs (int, optional): Sampling frequency. Default is 2048.
        ax (plt.Axes, optional): Axes to plot the PSD. Default is None.
        palette_name (str, optional): Name of the seaborn palette used for plotting.
            Default is "viridis".
        log_scale (bool, optional): If True, the PSD is plotted in log scale. Default
            is False.
        **kwarg: Additional keyword arguments to be passed to the `plot` function of
            matplotlib.

    Returns:
        plt.Axes: Axes where the PSD is plotted.
    """

    # Initialise variables
    chs = data.shape[1]

    # Define axis if None
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10,5), layout='tight')

    # Create color palette and assign it to the axis cycle
    color_palette = sns.color_palette(palette_name, chs).as_hex()
    ax.set_prop_cycle(color=color_palette)

    # Plot PSD
    psd, xf = get_psd(data, fs)

    if log_scale:
        ax.semilogy(xf, psd, **kwarg)
        ax.set_ylabel('PSD (dB/Hz)')
    else:
        ax.plot(xf, psd, **kwarg)
        ax.set_ylabel('PSD (mV^2/Hz)')
    ax.set_xlabel('Frequency (Hz)')

    return ax

def plot_psd_map(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    ax: Optional[plt.Axes] = None,
    log_scale: Optional[bool] = False,
    eps: Optional[float] = None,
    palette_name: Optional[str] = "viridis",
    **kwarg: Optional[Union[str, int, float]]
) -> plt.Axes:

    """Plot the Power Spectral Density (PSD) of the input data as a heatmap across
    channels.  

    Args:
        data (np.ndarray): Input data with shape (samples, channels).
        fs (int, optional): Sampling frequency. Default is 2048.
        ax (plt.Axes, optional): Axes to plot the PSD. Default is None.
        log_scale (bool, optional): If True, the PSD is plotted in log scale. Default
            is False.
        eps (float, optional): Small value to avoid log(0) when plotting in log scale.
            Default is None.
        palette_name (str, optional): Name of the seaborn palette used for plotting.
            Default is "viridis".
        **kwarg: Additional keyword arguments to be passed to the `plot` function of
            matplotlib.

    Returns:
        plt.Axes: Axes where the PSD is plotted.
    """

    # Initialise variables
    samples, chs = data.shape

    # Plot PSD
    psd, xf = get_psd(data, fs)

    # Plot PSD
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10,5), layout='tight')
    if log_scale:
        if eps is None: eps = psd.min()
        norm=colors.LogNorm(vmin=eps, vmax=psd.max())
        cbar_label = 'PSD (dB/Hz)'
    else:
        norm=None
        cbar_label = 'PSD (mV^2/Hz)'
    im = ax.pcolormesh(
        np.arange(chs), xf, psd, 
        shading='gouraud', cmap=palette_name, norm=norm,
        **kwarg
        )
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=90)
    ax.grid(False)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Channels')

    return ax


def plot_comp_spectrogram(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    ax: Optional[plt.Axes] = None, 
    log_scale: Optional[bool] = False,
    eps: Optional[float] = None,
    palette_name: Optional[str] = "viridis",
    **kwarg: Optional[Union[str, int, float]]   
) -> plt.Axes:

    """Compute and plot the spectrogram of the input data.

    Args:
        data (np.ndarray): Input data with shape (samples, channels).
        fs (int, optional): Sampling frequency. Default is 2048.
        ax (plt.Axes, optional): Axes to plot the spectrogram. Default is None.
        log_scale (bool, optional): If True, the PSD is plotted in log scale. Default
            is False.
        eps (float, optional): Small value to avoid log(0) when plotting in log scale.
            Default is None.
        palette_name (str, optional): Name of the seaborn palette used for plotting.
            Default is "viridis".
        **kwarg: Additional keyword arguments to be passed to the `pcolormesh` 
            function of matplotlib.
    Returns:
        plt.Axes: Axes where the spectrogram is plotted.
    """

    # Compute spectrogram (one sided)
    f, t, sxx = signal.spectrogram(data, fs)

    # Define axis if None
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10,5), layout='tight')

    # Normalise spectrogram if log scale
    if log_scale:
        if eps is None: eps = sxx.min()
        norm=colors.LogNorm(vmin=eps, vmax=sxx.max())
        cbar_label = 'PSD (dB/Hz)'
    else:
        norm=None
        cbar_label = 'PSD (mV^2/Hz)'

    # Plot spectrogram
    im = ax.pcolormesh(
        t, f, sxx,
        shading='gouraud', cmap=palette_name, norm=norm,
        **kwarg
    )
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=90)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')

    return ax

def plot_spectrogram(
    f: np.ndarray,
    t: np.ndarray,
    sxx: np.ndarray, 
    ax: Optional[plt.Axes] = None,
    palette_name: Optional[str] = "magma",
    log_scale: Optional[bool] = False,
    eps: Optional[float] = None,
    **kwarg: Optional[Union[str, int, float]]   
) -> plt.Axes:

    """Plot the input spectrogram.

    Args:
        f (np.ndarray): Frequencies of the spectrogram.
        t (np.ndarray): Timestamps of the spectrogram.
        sxx (np.ndarray): Spectrogram of one or more signals with shape
            (frequencies, timestamps, signals).
        ax (plt.Axes, optional): Axes to plot the spectrogram. Default is None.
        norm (bool, optional): If True, the PSD is plotted in log scale. Default
            is False.
        palette_name (str, optional): Name of the seaborn palette used for plotting.
            Default is "magma".
        log_scale (bool, optional): If True, the PSD is plotted in log scale. Default
            is False.
        eps (float, optional): Small value to avoid log(0) when plotting in log scale.
            Default is None.
        **kwarg: Additional keyword arguments to be passed to the `pcolormesh` 
            function of matplotlib.
    Returns:
        plt.Axes: Axes where the spectrogram is plotted.
    """

    # Normalise spectrogram if log scale
    if eps is None:
        eps = sxx.min()
    if log_scale:
        norm=colors.LogNorm(vmin=eps, vmax=sxx.max())
        cbar_label = 'PSD (dB/Hz)'
    else:
        norm=colors.Normalize(vmin=eps, vmax=sxx.max())
        cbar_label = 'PSD (mV^2/Hz)'

    # Plot spectrogram per signal
    if len(sxx.shape) == 2:
        sxx = np.expand_dims(sxx, axis=-1)
    nsig = sxx.shape[-1]

    if ax is None:
        cols = 1
        rows = np.round(nsig/cols).astype(int)

        fig, ax = plt.subplots(
            rows, cols, figsize=(12, 6),
            sharex=True, sharey=True, layout='constrained'
        ) 
        ax = ax.flatten()

    for i in range(nsig):
        # Plot spectrogram
        im = ax[i].pcolormesh(
            t, f, sxx[:,:,i],
            shading='gouraud', cmap=palette_name, norm=norm,
        )
        cbar = ax[i].figure.colorbar(im, ax=ax[i])
        cbar.set_label(cbar_label, rotation=90)
        ax[i].set_ylabel('Frequency (Hz)')
        ax[i].set_xlabel('Time (s)')
        ax[i].set_title(f'Channel {i}')

    return ax