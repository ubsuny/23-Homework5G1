from cmath import exp, pi
from math import sin, cos
from scipy.signal import find_peaks
import numpy as np

def discrete_transform(data):
    """
    Return the Discrete Fourier Transform (DFT) of a complex data vector to readable.

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
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    return np.array([even[k] + np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)] +
                    [even[k] - np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)])



def plot_power_spectrum_with_peaks(data, frequencies, sampling_rate=1):
    """
    Plot the power spectrum of the given data and mark the peaks.

    Parameters:
    - data (array-like): Time-domain signal for which the FFT is to be calculated.
    - frequencies (array-like): Frequencies corresponding to the FFT.
    - sampling_rate (float, optional): Sampling rate of the signal. Default is 1.

    Returns:
    - peak_frequencies (array): Frequencies of the identified peaks.

    This function calculates the power spectrum using FFT, identifies peaks,
    and plots the power spectrum with marked peaks.

    Example usage:
    ```
    from fft import plot_power_spectrum_with_peaks

    # Assuming you have cut_data and frequencies
    plot_power_spectrum_with_peaks(cut_data, frequencies)
    ```
    """
    # Perform FFT on the data
    X = fft(cut_data)

    # Calculate power spectrum
    power_spectrum = np.abs(data[:len(frequencies)//2])**2 / len(data)

    # Find peaks in the power spectrum
    peak_indices, _ = find_peaks(power_spectrum)

    # Get corresponding frequencies of the peaks
    peak_frequencies = frequencies[peak_indices]

    # Plot the power spectrum with identified peaks
    plt.plot(frequencies[:len(frequencies)//2], power_spectrum, label='Power Spectrum')
    plt.plot(peak_frequencies, power_spectrum[peak_indices], 'rx', label='Peaks')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.legend()
    plt.show()

    # Return the identified peak frequencies
    return peak_frequencies


def fft_power(x):
    """
    Compute the power spectrum of the given FFT result.

    Parameters:
    - x (array-like): Result of the FFT.

    Returns:
    array: Power spectrum of the input.

    The function calculates the power spectrum by squaring the magnitude
    of each frequency component in the FFT result and normalizing by the
    total number of data points.

    Note: The input should be the result of an FFT, and the length of the
    input should be a power of 2 for optimal efficiency.
    """
    N = len(x)
    if N <= 1:
        return x

    power = np.zeros(N // 2 + 1)
    power[0] = abs(x[0]) ** 2
    power[1 : N // 2] = abs(x[1 : N // 2]) ** 2 + abs(x[N - 1 : N // 2 : -1]) ** 2
    power[N // 2] = abs(x[N // 2]) ** 2
    power = power / N

    return power

def ifft(x):
    from numpy import conj, divide

    # Conjugate the complex numbers
    x = np.conj(x)

    # Forward FFT
    X = fft(x)

    # Conjugate the complex numbers again
    X = np.conj(X)

    # Scale the numbers
    X = X / len(X)

    # Perform the inverse FFT
    result = np.fft.ifft(X)

    return result



