# Plotting the peak frequency spectrum
plt.figure(figsize=(10, 5))
plt.plot(fft(Heavyside[:128]*np.abs(X[:128])))
plt.title('Peak Frequency Spectrum of Monthly Average CO2 Concentration', fontsize=16, fontweight='bold', color='blue')
plt.xlabel('Frequency (cycles per month)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
