from fft import * 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Flag for applying window function
window = True

# Load CO2 data from NOAA, skip header rows, and name columns
df = pd.read_csv(
    'https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/co2_mid_surface-flask_1_ccgg_month.txt',
    delimiter="\s+", skiprows=54, names=['site', 'year', 'month', 'value'])

# Extract CO2 concentration values and filter out invalid (negative) values
y = df['value'].values
y_valid = y >= 0.
y = y[y_valid]

# Calculate the length of the valid data
M = len(y)
# Determine the next power of 2 greater than the number of data points
log2M = math.log(M, 2)
next_pow_of_2 = int(log2M) + 1

# Pad data with zeros to match the next power of 2, if needed
if log2M - int(log2M) > 0.0:
    ypads = np.full(2**(next_pow_of_2) - M, 0, dtype=np.double)
    y = np.concatenate((y, ypads))
    # Create an index array for the padded data
    x = np.arange(len(y))
    M = len(y)
    # Apply a window function to reduce spectral leakage
    if window:
        y = y * (0.5 - 0.5 * np.cos(2 * np.pi * x / (M - 1)))

# Perform FFT on the data
Y = fft(y)
# Calculate the magnitude and power of the Fourier components
Y_abs = abs(Y)
powery = fft_power(Y) 
powerx = np.arange(powery.size)

# Plotting the original CO2 concentration data
f1 = plt.figure(1)
plt.plot(x, y)
plt.xlabel("Index")
plt.ylabel("CO$_2$ Concentration")

# Plotting the Fourier components in terms of power and magnitude
f2 = plt.figure(2)
plt.plot(powerx, powery, label="Power")
plt.plot(x, Y_abs, label="Magnitude")
plt.xlim([0, M / 4])
plt.legend()
plt.yscale('log')
plt.xlabel("Spectral Index")
plt.ylabel("Fourier Component")

plt.show()
