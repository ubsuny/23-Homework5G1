import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    'https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/co2_mid_surface-flask_1_ccgg_month.txt',
    delimiter="\s+",skiprows=54, names=['site',	'year',	'month',	'value'])

def process_fft(data, time_step=1, threshold=0.01):
    """
    Process data using FFT and return the cleaned data.
    """
    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(fft_result), d=time_step)
    fft_result[np.abs(frequencies) > threshold] = 0
    return np.fft.ifft(fft_result)

def plot_data_comparison(raw_data, cleaned_data, title, xlabel, ylabel):
    """
    Plot both raw and cleaned data for comparison.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(raw_data, label='Raw Data')
    plt.plot(cleaned_data.real, label='Cleaned Data', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# FFT processing
cleaned_data = process_fft(df['value'])

# Plotting raw data and cleaned data for comparison
plot_data_comparison(df['value'], cleaned_data, 'CO2 Concentration Comparison', 'Time (Months)', 'CO2 Concentration')
