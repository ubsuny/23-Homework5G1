from fft import *
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    'https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/co2_mid_surface-flask_1_ccgg_month.txt',
    delimiter="\s+",skiprows=54, names=['site',	'year',	'month',	'value'])

# Read like previous example with CO2 data
y = df['value'].values
y_valid = y >= 0.
y = y[y_valid]

# instead of truncating, pad with values

M = len(y)
log2M = math.log(M, 2)
next_pow_of_2 = int(log2M) + 1
if log2M - int(log2M) > 0.0 :    
    ypads = np.full( 2**( next_pow_of_2) - M, 0, dtype=np.double)
    y = np.concatenate( (y, ypads) )
    # CAREFUL: When you pad, the x axis becomes somewhat "meaningless" for the padded values, 
    # so typically it is best to just consider it an index
    x = np.arange(len(y))
    M = len(y)
                
# Get the FFT
Y = fft(y)
# Smooth the data in the Fourier domain.
# Adjust this to change the frequencies to delete (frequencies are removed from maxfreq to N/2
# and accounts for the Nyquist frequency). 
maxfreq = 50
Y[maxfreq:len(Y)-maxfreq] = 0.0
# Get the absolute value and power for plotting
Y_abs = abs(Y)
powery = fft_power(Y)
powerx = np.arange(powery.size)

# Now go back to the frequency domain. 
# Compare the data before and after filtering. 
yfiltered = ifft(Y)
yfiltered_abs= abs(yfiltered)


f2 = plt.figure(2)
plt.plot( powerx, powery, label="Power" )
plt.plot( x, Y_abs, label="Magnitude" )
plt.legend()
plt.xlim([0,maxfreq*2])
plt.yscale('log')
plt.xlabel("Spectral Index")
plt.ylabel("Fourier Component")

plt.show()
