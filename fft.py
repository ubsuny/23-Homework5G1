from cmath import exp, pi
from math import sin, cos
import numpy as np

def discrete_transform(data):
    """
    Return the Discrete Fourier Transform (DFT) of a complex data vector.

    This function calculates the DFT of a given complex data vector using the formula:
    F(k) = Σ [data(j) * exp(-2πi * k * j / N)] for j in range(N)

    Parameters:
    - data (array-like): A complex data vector for which the DFT is to be calculated.

    Returns:
    array: The complex DFT of the input data vector.
    """
    N = len(data)
    transform = np.zeros(N, dtype=np.complex128)

    for k in range(N):
        for j in range(N):
            angle = 2 * pi * k * j / N
            transform[k] += data[j] * exp(1j * angle)

    return transform
