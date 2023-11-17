import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TestFFTVisualization(unittest.TestCase):

    def setUp(self):
        # Assuming df is a DataFrame containing the 'value' column
        self.df = pd.DataFrame({'value': np.random.rand(100) * 100})

    def test_fft_visualization(self):
        window = True

        # Extract 'value' column from the DataFrame
        y = self.df['value'].values
        y_valid = y >= 0.
        y = y[y_valid]

        # Pad with values
        M = len(y)
        log2M = int(np.log2(M))
        next_pow_of_2 = log2M + 1
        if log2M - int(log2M) > 0.0:
            ypads = np.full(2**next_pow_of_2 - M, 0, dtype=np.double)
            y = np.concatenate((y, ypads))
            x = np.arange(len(y))
            M = len(y)

            # Apply a window to reduce ringing from the 2^n cutoff
            if window:
                y = y * (0.5 - 0.5 * np.cos(2 * np.pi * x / (M - 1)))

        # Get the FFT
        Y = fft(y)
        Y_abs = abs(Y)
        powery = fft_power(Y)
        powerx = np.arange(powery.size)

        # Plot the results
        f1 = plt.figure(1)
        plt.plot(x, y)
        plt.xlabel("Index")
        plt.ylabel("CO$_2$ Concentration")

        f2 = plt.figure(2)
        plt.plot(powerx, powery, label="Power")
        plt.plot(x, Y_abs, label="Magnitude")
        plt.xlim([0, M / 4])
        plt.legend()
        plt.yscale('log')
        plt.xlabel("Spectral Index")
        plt.ylabel("Fourier Component")

        # Additional assertions based on your specific expectations
        self.assertEqual(len(x), M)
        self.assertEqual(len(Y), M)

        # Display the plots (comment out if running in non-interactive mode)
        plt.show()

if __name__ == '__main__':
    unittest.main()
