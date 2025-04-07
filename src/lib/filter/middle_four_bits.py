"""
Middle Four Bits Filter implementation.
"""
import numpy as np
from abc import ABC, abstractmethod

class Filter(ABC):
    """
    Abstract base class for all filters.

    A filter takes an input frame, applies a filtering operation, and returns a filtered frame.
    """

    @abstractmethod
    def filter(self, frame):
        """
        Filter a frame.

        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).

        Returns:
            filtered_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        pass

    def validate_frame(self, frame, expected_shape=None, expected_dtype=np.uint8):
        """
        Validate that the frame has the expected shape and data type.

        Args:
            frame: NumPy array to validate.
            expected_shape: Tuple of expected shape (height, width, channels) or None.
            expected_dtype: Expected data type of the frame.

        Returns:
            frame: The validated frame, converted to the expected data type if necessary.

        Raises:
            ValueError: If the frame does not have the expected shape.
        """
        # Ensure the frame is a NumPy array
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a NumPy array")

        # Ensure the frame has the expected shape
        if expected_shape is not None:
            if frame.shape != expected_shape:
                raise ValueError(f"Frame must have shape {expected_shape}, got {frame.shape}")

        # Ensure the frame has the expected data type
        if frame.dtype != expected_dtype:
            frame = frame.astype(expected_dtype)

        return frame

class MiddleFourBitsFilter(Filter):
    """
    Filter that preserves only the middle four bits (bits 2-5) of each pixel in an RGB frame,
    setting all other bits (0, 1, 6, 7) to zero.
    """

    def filter(self, frame):
        """
        Filter a frame by preserving only the middle four bits.

        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).

        Returns:
            filtered_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)

        # Create a mask for bits 2-5: 00111100 in binary, which is 60 in decimal
        mask = 0b00111100  # Decimal 60, preserves bits 2, 3, 4, 5

        # Apply the mask to all pixels in all channels using bitwise AND
        # This zeros out bits 0, 1, 6, and 7, keeping bits 2-5 unchanged
        filtered_frame = frame & mask

        return filtered_frame
