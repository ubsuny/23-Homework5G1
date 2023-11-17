from fft import *
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
window = True

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
    # Apply a window to reduce ringing from the 2^n cutoff
    if window : 
        y = y * (0.5 - 0.5 * np.cos(2*np.pi*x/(M-1)))
                
Y = fft(y)
Y_abs = abs(Y)
powery = fft_power(Y)
powerx = np.arange(powery.size)

f1 = plt.figure(1)
plt.plot( x, y )
plt.xlabel("Index")
plt.ylabel("CO$_2$ Concentration")

f2 = plt.figure(2)
plt.plot( powerx, powery, label="Power" )
plt.plot( x, Y_abs, label="Magnitude" )
plt.xlim([0,M/4])
plt.legend()
plt.yscale('log')
plt.xlabel("Spectral Index")
plt.ylabel("Fourier Component")

plt.show()
