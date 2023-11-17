# Importing necessary libraries
from fft import *  
import math      
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Reading CO2 data from the NOAA website
df = pd.read_csv(
    'https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/co2_mid_surface-flask_1_ccgg_month.txt',
    delimiter="\s+",skiprows=54, names=['site',	'year',	'month',	'value'])
# Extracts CO2 values, filters invalid ones, and stores them in 'y'
y = df['value'].values
y_valid = y >= 0.
y = y[y_valid]

# Padding the data for FFT
M = len(y)
log2M = math.log(M, 2)
next_pow_of_2 = int(log2M) + 1
if log2M - int(log2M) > 0.0 :    
    ypads = np.full( 2**( next_pow_of_2) - M, 0, dtype=np.double)
    y = np.concatenate( (y, ypads) )
    x = np.arange(len(y))
    M = len(y)
                
# Performing FFT on the data
Y = fft(y)

# Smoothing the data by manipulating frequencies in the Fourier domain
maxfreq = 50
Y[maxfreq:len(Y)-maxfreq] = 0.0

# Computing absolute values and power of the Fourier transform for plotting
Y_abs = abs(Y)
powery = fft_power(Y)
powerx = np.arange(powery.size)

# Inverse FFT to get filtered time-domain signal
yfiltered = ifft(Y)
yfiltered_abs= abs(yfiltered)

# Plotting the results
f2 = plt.figure(2)
plt.plot( powerx, powery, label="Power" )
plt.plot( x, Y_abs, label="Magnitude" )
plt.legend()
plt.xlim([0,maxfreq*2])
plt.yscale('log')
plt.xlabel("Spectral Index")
plt.ylabel("Fourier Component")

plt.show()
