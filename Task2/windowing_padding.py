import numpy as np
import matplotlib.pyplot as plt
# Padding
log2N = np.log2(len(y))
next_pow_of_2 = int(log2N) + 1
if log2N != int(log2N):
    y_padded = np.pad(y, (0, 2**next_pow_of_2 - len(y)), 'constant', constant_values=(0,))
else:
    y_padded = y
x_padded = np.arange(len(y_padded))

# Windowing
window = 0.5 - 0.5 * np.cos(2 * np.pi * x_padded / (len(y_padded) - 1))
y_windowed = y_padded * window

# FFT
Y = fft(y_windowed)

# Filtering (e.g., remove high frequencies)
maxfreq = 5
Y_filtered = Y.copy()
Y_filtered[maxfreq:len(Y)-maxfreq] = np.zeros(len(Y)-2*maxfreq)

# Inverse FFT
y_filtered = ifft(Y_filtered)
y_filtered_abs = np.abs(y_filtered)

# Plotting
# CO2 concentration plot
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(y_windowed, label='windowed data')
plt.plot(y[:len(y)], label='Original')  # Plotting only the original length part
plt.xlabel("Month Index")
plt.ylabel("CO2 Concentration (ppm)")
plt.legend()
