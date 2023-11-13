def calculate_frequencies(X):
    sample_spacing = 1  # assuming monthly data, so the spacing is 1 month
    N = len(X)
    freqs = np.fft.fftfreq(N, d=sample_spacing)
    return freqs[:N//2]  # return only the positive frequencies

def identify_peak_frequency(frequencies, X):
    # Find the index of the peak frequency
    peak_index = np.argmax(np.abs(X[:len(frequencies)]))
    return frequencies[peak_index]

# Calculate frequencies
frequencies = calculate_frequencies(X)

# Find the peak frequency
peak_frequency = identify_peak_frequency(frequencies, X)
