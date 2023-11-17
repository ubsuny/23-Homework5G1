"""
Module containing functions for frequency calculation and identification.
"""

import unittest
import numpy as np

def calculate_frequencies(X):
    """
    Calculate the frequencies for the given data.

    Parameters:
    - X (numpy.ndarray): Input data.

    Returns:
    - numpy.ndarray: Array of frequencies.
    """
    sample_spacing = 1  # assuming monthly data, so the spacing is 1 month
    N = len(X)
    freqs = np.fft.fftfreq(N, d=sample_spacing)
    return freqs[:N//2]  # return only the positive frequencies

def identify_peak_frequency(frequencies, X):
    """
    Identify the peak frequency from the given frequencies and signal.

    Parameters:
    - frequencies (numpy.ndarray): Array of frequencies.
    - X (numpy.ndarray): Input signal.

    Returns:
    - float: Peak frequency.
    """
    peak_index = np.argmax(np.abs(X[:len(frequencies)]))
    return frequencies[peak_index]

class TestFrequencyIdentification(unittest.TestCase):
    """
    Unit tests for frequency identification functions.
    """
    def test_calculate_frequencies(self):
        """
        Test the calculate_frequencies function.
        """
        test_data = np.array([2, 4, 6, 8, 10])  # Example data
        expected_freqs = np.fft.fftfreq(len(test_data), d=1)[:len(test_data)//2]
        calculated_freqs = calculate_frequencies(test_data)
        self.assertTrue(np.array_equal(expected_freqs, calculated_freqs))

    def test_identify_peak_frequency(self):
        """
        Test the identify_peak_frequency function.
        """
        test_freqs = np.array([1, 2, 3, 4, 5])
        test_signal = np.array([10, 20, 15, 30, 25])
        expected_peak_freq = test_freqs[np.argmax(np.abs(test_signal[:len(test_freqs)]))]
        calculated_peak_freq = identify_peak_frequency(test_freqs, test_signal)
        self.assertEqual(expected_peak_freq, calculated_peak_freq)
if __name__ == '__main__':
    unittest.main()
