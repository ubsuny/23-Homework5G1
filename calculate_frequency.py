import numpy as np

def calculate_frequencies(X, sample_spacing=1):
    """
    Calculate the frequencies corresponding to the FFT output.

    Parameters:
    - X (array-like): The output of the FFT.
    - sample_spacing (float, optional): The spacing between data points. Defaults to 1 (assuming monthly data).

    Returns:
    - array-like: Frequencies corresponding to the FFT output.
    """
    N = len(X)
    freqs = np.fft.fftfreq(N, d=sample_spacing)
    return freqs[:N//2]  # return only the positive frequencies

def identify_peak_frequency(frequencies, X):
    """
    Identify the peak frequency from the FFT output.

    Parameters:
    - frequencies (array-like): Frequencies corresponding to the FFT output.
    - X (array-like): The output of the FFT.

    Returns:
    - float: The peak frequency.
    """
    # Find the index of the peak frequency
    peak_index = np.argmax(np.abs(X[:len(frequencies)]))
    return frequencies[peak_index]

# Calculate frequencies
frequencies = calculate_frequencies(X)

# Find the peak frequency
peak_frequency = identify_peak_frequency(frequencies, X)
