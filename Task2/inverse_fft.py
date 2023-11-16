import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(
    'https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/co2_mid_surface-flask_1_ccgg_month.txt',
    delimiter="\s+",skiprows=54, names=['site',	'year',	'month',	'value'])

# Extracting the CO2 concentration values
y = df['value'].values
# FFT and IFFT functions
def fft(x):
    """
    Perform Fast Fourier Transform (FFT) on the input signal.

    Parameters:
    - x (array-like): Input signal.

    Returns:
    - array-like: FFT result.
    """
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def ifft(x):
    """
    Perform Inverse Fast Fourier Transform (IFFT) on the input signal.

    Parameters:
    - x (array-like): Input signal.

    Returns:
    - array-like: IFFT result.
    """
    x_conj = np.conj(x)
    x_fft = fft(x_conj)
    x_fft_conj = np.conj(x_fft)
    return x_fft_conj / len(x_fft_conj)
