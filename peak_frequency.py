import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_peak_frequency(data):
    """
    Function takes a pandas DataFrame `data` containing a column named 'value' to represent a time series of measurements. 

    Parameters:
    - data : pandas DataFrame
        Data containing date and values with date assigned in years and months.

    Returns:
    - cycles_per_year : float
        The estimated frequency in cycles per year.
    """  
    values = data['value'].to_numpy()

    # Perform a Fast Fourier Transform (FFT) on the data
    timestep = 1  # Calculate the time step (assuming a regular monthly sampling)
    fft_result = np.fft.fft(values)
    fft_freq = np.fft.fftfreq(len(fft_result), d=timestep)  # Compute the corresponding frequencies in cycles per month

    # Set a threshold to identify high-frequency components
    threshold = 0.01
    fft_result[np.abs(fft_freq) > threshold] = 0  # Zero out high-frequency components
    cleaned_data = np.fft.ifft(fft_result)

    # Find the frequency component with the highest magnitude
    peak_frequency_index = np.argmax(np.abs(cleaned_data))
    peak_frequency = np.abs(fft_freq[peak_frequency_index])

    # Convert the peak frequency to useful units
    cycles_per_year = peak_frequency * 12  # Assuming 12 months in a year

    return cycles_per_year


data = pd.read_csv('https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/co2_mid_surface-flask_1_ccgg_month.txt',
                   delimiter='\s+', skiprows=54, names=['site', 'year', 'month', 'value'])
cycles_per_year = find_peak_frequency(data)
print("Peak Frequency (Cycles per year):", cycles_per_year)
