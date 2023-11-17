"""Unit test for windowing"""

import unittest
import numpy as np
from fft import fft  # assuming fft.py is in the same directory as this test file

class TestFFTFunction(unittest.TestCase):
    """
    A unit test class for the fft function in the fft module.
    """

    def test_fft(self):
        """
        Test docstring: Describe what this specific test is checking.
        """
        # Your test data
        input_data = [1, 2, 3, 4]
         # Expected output for the given test data
        expected_output = np.fft.fft(input_data)
        # Call the fft function from the module
        result = fft(input_data)
        np.testing.assert_allclose(result, expected_output)
if __name__ == '__main__':
    unittest.main()
