"""Tools for frequency domain analysis of EMG data"""

from typing import Optional, Tuple
from scipy import fft
import numpy as np

def get_psd(
    data: np.ndarray,
    fs: Optional[int] = 2048,
) -> Tuple[np.ndarray, np.ndarray]:

    """Compute the Power Spectral Density (PSD) of the input data.

    Args:
        data (np.ndarray): Input data with shape (samples, channels).
        fs (int, optional): Sampling frequency. Default is 2048.
        
    Returns:
        psd (np.ndarray): PSD of the input data with shape (samples//2, 
            channels), where the first dimension corresponds to the 
            positive frequencies.
        xf (np.ndarray): Frequency axis of the PSD with shape (samples//2,).
    """

    # Initialise variables
    samples, chs = data.shape

    # Compute PSD
    yf = fft.fft(data, axis=0)
    xf = fft.fftfreq(samples, 1/fs)[:samples//2]
    psd = 2/samples * np.abs(yf[0:samples//2])

    return psd, xf