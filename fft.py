from cmath import exp, pi
from math import sin, cos
import numpy as np

def discrete_transform(data):
    """
    Return the Discrete Fourier Transform (DFT) of a complex data vector.

    This function calculates the DFT of a given complex data vector using the formula:
    F(k) = Σ [data(j) * exp(-2πi * k * j / N)] for j in range(N)

    Parameters:
    - data (array-like): A complex data vector for which the DFT is to be calculated.

    Returns:
    array: The complex DFT of the input data vector.
    """
    N = len(data)
    transform = np.zeros(N, dtype=np.complex128)

    for k in range(N):
        for j in range(N):
            angle = 2 * pi * k * j / N
            transform[k] += data[j] * exp(1j * angle)

    return transform

def fft(x):
    """
    Perform the Cooley-Tukey Radix-2 Decimation in Time (DIT) Fast Fourier Transform (FFT).

    Parameters:
    - x (array-like): Input signal for which the FFT is to be calculated.

    Returns:
    array: Complex array representing the FFT of the input signal.

    The function uses a recursive approach based on the Cooley-Tukey algorithm
    to efficiently compute the FFT. If the length of the input signal is odd,
    it falls back to the discrete_transform function.

    Note: The input signal length should be a power of 2 for optimal efficiency.
    """
    def fft(x):    # use our y value from our plot/data as x here...
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    return np.array( [even[k] + np.exp(-2j*np.pi*k/N)*odd[k] for k in range(N//2)] + \
                     [even[k] - np.exp(-2j*np.pi*k/N)*odd[k] for k in range(N//2)] )
