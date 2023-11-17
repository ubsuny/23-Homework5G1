from fft import fft
import numpy as np
import pandas as pd

# Load data from a URL into a pandas DataFrame
url = 'https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/co2_mid_surface-flask_1_ccgg_month.txt'
df = pd.read_csv(url, delimiter="\s+", skiprows=54, names=['site', 'year', 'month', 'value'])


# Convert 'value' column to a NumPy array before slicing
cut_data = df['value'].to_numpy()[:256]
X = fft(cut_data)

def calculate_frequencies(X):
    sample_spacing = 1/12  #assuming yearly data, so the spacing is cycles per year
    N = len(X)
    freqs = np.fft.fftfreq(N, d=sample_spacing)
    return freqs[:N//2]   # return only positive frequencies
#calculate frequencies
frequencies = calculate_frequencies(X)

print(frequencies)