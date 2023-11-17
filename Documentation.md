# Documentation
## Introduction:
The FFT essentially converts a signal from its time-based representation to its frequency-based representation, allowing us to dissect the signal into its fundamental frequencies. This process has broad applications across multiple fields, such as signal processing, engineering, and physics. In the case of CO2 or methane data, applying FFT enables the understanding of recurring patterns or prominent frequencies in the dataset. It helps in uncovering cyclic behaviors or notable shifts in gas concentrations over time. By breaking down the data into its frequency components, the FFT reveals the waves needed to reconstruct the original signal. This is particularly valuable for recognizing cyclic trends, identifying significant frequencies, and eliminating unwanted noise from the dataset.
The Fourier Transform of a continuous-time signal \( x(t) \) is given by:

X(f) = ∫<sub>-∞</sub><sup>∞</sup> x(t) ⋅ e<sup>-j2πft</sup> dt

where:
- \( X(f) \) is the complex amplitude of the signal in the frequency domain at frequency \( f \).
- \( x(t) \) is the time-domain signal.
- \( j \) is the imaginary unit.

The Inverse Fourier Transform is used to recover the original time-domain signal from its frequency-domain representation. For a continuous-time signal:

X(t) = ∫<sub>-∞</sub><sup>∞</sup> x(f) ⋅ e<sup>-j2πft</sup> df

## Data Selection
[Data Station](https://gml.noaa.gov/dv/data/index.php?category=Greenhouse%2BGases&parameter_name=Carbon%2BDioxide&frequency=Monthly%2BAverages&search=sand+island): We have chosen the data of Sand Island, Midway, United States (MID) where air samples are collected in glass flasks.
## Project Objective:
The project is divided into three tasks.

#### Task 1: Calculate Actual Frequency
We have to implement a Python function for actual frequency in useful units based on the Fourier transform of a signal. The function should take frequencies and the Fourier transform of a signal, along with the sampling rate, and provide the actual frequency in a more interpretable form.

#### Task 2: Clean Up Noise in Time Domain
We have to develop a Python function that utilizes a combination of waveform modification techniques, including padding, windowing, FFT, inverse FFT, and undoing window+padding. This function should clean up either high or low-frequency noise in the time domain by zeroing the appropriate waveform coefficients in the frequency domain. Additionally, you need to plot both the "raw" and "cleaned" spectra in the time domain for visual comparison.

#### Task 3: Documentation, GitHub Actions, and Unit Tests
For this task, we have to complete the documentation for the implemented functions, reuse GitHub Actions for linting and unit tests, and write unit tests for Task 1 and Task 2.

## Algorithm with Docstrings.
```python
from cmath import exp, pi
from math import sin, cos
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

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
    X = fft(data)

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
    """
    Compute the Inverse Fast Fourier Transform (IFFT) of the given signal.

    Parameters:
    - x (array-like): Result of the FFT.

    Returns:
    array: Time-domain signal after IFFT.

    The function calculates the IFFT by conjugating the complex numbers,
    performing the forward FFT, conjugating the complex numbers again,
    scaling the numbers, and then performing the inverse FFT.
    """
    from numpy import conj

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
```
We got the values of frequencies as mentioned below:
```python
Peak Frequencies: [0.31858407 0.63716814 0.95575221 1.24778761 1.56637168 1.91150442
 2.2300885  2.54867257 2.86725664 3.18584071 3.50442478 3.79646018
 4.14159292 4.3539823  4.46017699 4.67256637 4.77876106 5.09734513
 5.4159292  5.73451327]
```
## Filtering

Filtering refers to the process of selectively modifying or extracting certain components from a signal while attenuating or eliminating others. It is a fundamental concept in signal processing and is used in various fields such as audio processing, image processing, communication systems, and data analysis. The goal of filtering is often to enhance specific features of interest or to remove unwanted noise or interference.

#### Types of Filters

1. **Low-pass Filter:**
   - Allows low-frequency components to pass through while attenuating high-frequency components.
   - Commonly used to smooth or remove high-frequency noise from a signal.

2. **High-pass Filter:**
   - Allows high-frequency components to pass through while attenuating low-frequency components.
   - Useful for extracting or enhancing high-frequency features in a signal.

#### Applications of Filtering

1. **Audio Processing:** Filtering is used in equalization to adjust the balance of different frequency components in audio signals.
2. **Image Processing:** Filtering is applied to enhance or suppress certain features in images.
3. **Communication Systems:** Filters are used to extract or eliminate specific frequency bands in signals for modulation, demodulation, and noise reduction.
4. **Biomedical Signal Processing:** Filtering is employed to remove noise and isolate specific physiological components from biomedical signals.

We have used high pass filter in our case to get the desired data. The graph is shown below:

![Plot after filtration](https://github.com/ubsuny/23-Homework5G1/blob/main/Task2/Plots/filtering.png)

## Windowing
Windowing is a technique used in signal processing to minimize the impact of discontinuities and artifacts that can occur when analyzing a finite segment (window) of a signal. It involves multiplying a signal by a window function, which is a mathematical function that is non-zero for only a finite interval.
#### Main Goals of Windowing

The application of windowing in signal processing serves several important goals:

1. **Minimize Spectral Leakage:**
   - Windowing helps mitigate the impact of spectral leakage, which occurs when the frequency content of a signal extends beyond the boundaries of the analysis window. By tapering the signal at the edges, windowing reduces the leakage of energy into neighboring frequency bins.

2. **Reduce Side Lobe Amplitude:**
   - Another crucial goal of windowing is to reduce the amplitude of side lobes. Side lobes are undesired peaks in the frequency domain that can occur when analyzing a finite segment of a signal. Window functions, such as Hamming, Hanning, or Blackman, are designed to suppress side lobes, improving the accuracy of spectral analysis.

3. **Improve Frequency and Amplitude Estimation Accuracy:**
   - Windowing contributes to the accuracy of frequency and amplitude estimation. By minimizing spectral leakage and suppressing side lobes, windowing enhances the precision with which frequency and amplitude information can be extracted from the signal. This is particularly important in applications where accurate signal characterization is critical.

In summary, windowing is a fundamental technique aimed at refining the analysis of signals, ensuring more accurate and reliable results in various signal processing applications.

The obtained nature of graph after windowing is given below:
![Plot after filtration](https://github.com/ubsuny/23-Homework5G1/blob/main/Task2/Plots/windowing1.png)

## Raw and cleaned Graph:
![Plot after filtration](https://github.com/ubsuny/23-Homework5G1/blob/main/Task2/Plots/raw_and_cleaned.png)

In this way we have cleaned the raw data and obtained the graph.




