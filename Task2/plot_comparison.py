# Importing necessary libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 

# Reading CO2 data from the NOAA website
df = pd.read_csv(
    'https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/co2_mid_surface-flask_1_ccgg_month.txt',
    delimiter="\s+", skiprows=54, names=['site', 'year', 'month', 'value'])

def process_fft(data, time_step=1, threshold=0.01):
    """
    Processes the given data using Fast Fourier Transform (FFT) to remove high-frequency noise.
    Parameters:
        data (array): The input data to be processed.
        time_step (float): The time interval between data points.
        threshold (float): The frequency threshold for filtering.
    Returns:
        ndarray: The cleaned data after inverse FFT.
    """
    fft_result = np.fft.fft(data) # Compute the FFT of the input data
    frequencies = np.fft.fftfreq(len(fft_result), d=time_step) # Calculate the corresponding frequencies
    fft_result[np.abs(frequencies) > threshold] = 0 # Zero out components with frequencies higher than the threshold
    return np.fft.ifft(fft_result) # Perform inverse FFT to convert back to time domain

def plot_data_comparison(raw_data, cleaned_data, title, xlabel, ylabel):
    """
    Plots a comparison between raw and cleaned data.
    Parameters:
        raw_data (array): The original, unprocessed data.
        cleaned_data (array): The data after processing.
        title (str): The title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(8, 5)) # Set the size of the plot
    plt.plot(raw_data, label='Raw Data') # Plot the raw data
    plt.plot(cleaned_data.real, label='Cleaned Data', linestyle='--') # Plot the cleaned data (real part)
    plt.title(title) # Set the title of the plot
    plt.xlabel(xlabel) # Set the label for the x-axis
    plt.ylabel(ylabel) # Set the label for the y-axis
    plt.legend() # Show the legend
    plt.show() # Display the plot

# Applying FFT processing to clean the data
cleaned_data = process_fft(df['value'])

# Plotting both the raw and the cleaned data for comparison
plot_data_comparison(df['value'], cleaned_data, 'CO2 Concentration Comparison', 'Time (Months)', 'CO2 Concentration')
