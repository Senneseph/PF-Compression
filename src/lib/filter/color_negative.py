"""
Color Negative Filter implementation.
"""
import numpy as np
from src.lib.filter.middle_four_bits import Filter

class ColorNegativeFilter(Filter):
    """
    Filter that converts an RGB frame into its color negative by inverting each channel.
    For each pixel value in each channel, the new value is 255 - original_value.
    """

    def filter(self, frame):
        """
        Filter a frame by converting it to its color negative.

        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).

        Returns:
            filtered_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)

        # Invert the frame: new_value = 255 - original_value for each channel
        negative_frame = 255 - frame

        return negative_frame
