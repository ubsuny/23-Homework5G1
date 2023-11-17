import unittest
import numpy as np
import pandas as pd

class TestFFTFiltering(unittest.TestCase):

    def setUp(self):
        # Load CO2 data from the given URL
        data_url = 'https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/co2_mid_surface-flask_1_ccgg_month.txt'
        data = pd.read_csv(data_url, delimiter='\s+', skiprows=54, names=['site', 'year', 'month', 'value'])

        # Extract the 'value' column and filter out negative values
        y_valid = data['value'] >= 0.
        self.y = data.loc[y_valid, 'value'].to_numpy()

        # Pad the data
        N = len(self.y)
        log2N = int(np.log2(N))
        next_pow_of_2 = log2N + 1
        if log2N - int(log2N) > 0.0:
            ypads = np.full(2**next_pow_of_2 - N, 0, dtype=np.double)
            self.y = np.concatenate((self.y, ypads))
            self.x = np.arange(len(self.y))
            N = len(self.y)

        # Set the maximum frequency to filter
        self.maxfreq = 50

    def test_fft_filtering(self):
        # Get the FFT
        Y = fft(self.y)

        # Smooth the data in the Fourier domain
        Y[self.maxfreq:len(Y)-self.maxfreq] = 0.0

        # Get the absolute value and power for plotting
        Y_abs = abs(Y)
        powery = fft_power(Y)
        powerx = np.arange(powery.size)

        # Now go back to the time domain
        yfiltered = ifft(Y)
        yfiltered_abs = abs(yfiltered)

        # Add assertions based on the expected behavior
        self.assertEqual(len(Y), len(self.y))
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()
