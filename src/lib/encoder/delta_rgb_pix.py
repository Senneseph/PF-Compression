"""
Delta RGB Pixel Encoder implementation.
"""
import numpy as np
from abc import ABC, abstractmethod

class Encoder(ABC):
    """
    Abstract base class for all encoders.

    An encoder takes an input frame and transforms it into a different representation.
    """

    @abstractmethod
    def encode(self, frame):
        """
        Encode a frame into a different representation.

        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).

        Returns:
            Encoded data in a format specific to the encoder implementation.
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

class DeltaRGBPixEncoder(Encoder):
    """
    Encoder that identifies changed RGB values compared to a persistent frame.
    """

    def __init__(self, persistent_frame=None, frame_shape=(480, 640, 3)):
        """
        Initialize the Delta RGB Pixel Encoder.

        Args:
            persistent_frame: Initial persistent frame. If None, a black frame is created.
            frame_shape: Shape of the frame (height, width, channels).
        """
        self.frame_shape = frame_shape
        if persistent_frame is None:
            self.persistent_frame = np.zeros(frame_shape, dtype=np.uint8)
        else:
            self.persistent_frame = persistent_frame.copy()

    def encode(self, frame):
        """
        Encode the frame by identifying changed RGB values compared to the persistent frame.

        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).

        Returns:
            tuple: (mask, new_values)
                - mask: 2D array of 3-bit values (0-7) indicating which channels changed.
                - new_values: 3D array of new RGB values for the changed channels.
        """
        frame = self.validate_frame(frame, expected_shape=self.frame_shape)

        # Compare the current frame with the persistent frame
        changed = frame != self.persistent_frame  # Shape: (height, width, 3), dtype: bool

        # Create a 3-bit mask indicating which channels changed
        # Bit 2 (4): R channel changed
        # Bit 1 (2): G channel changed
        # Bit 0 (1): B channel changed
        mask = (changed[:, :, 0].astype(np.uint8) * 4 +
                changed[:, :, 1].astype(np.uint8) * 2 +
                changed[:, :, 2].astype(np.uint8) * 1)

        # Create a 3D array of new RGB values for the changed channels
        new_values = np.zeros_like(frame)
        new_values[changed] = frame[changed]

        # Update the persistent frame
        self.persistent_frame = frame.copy()

        return mask, new_values

    def denoise(self, mask, new_values, frame, threshold=1):
        """
        Denoise the output of encode by excluding RGB changes below a threshold.

        Args:
            mask: 2D array of 3-bit values (0-7) indicating which channels changed.
            new_values: 3D array of new RGB values for the changed channels.
            frame: The original input frame.
            threshold: The maximum allowed difference to consider a change as noise (default: 1).

        Returns:
            tuple: (updated_mask, new_values)
                - updated_mask: 2D array with small changes excluded.
                - new_values: Original new_values (unchanged).
        """
        # Compute the absolute difference between the frame and persistent frame
        diff = np.abs(frame.astype(np.int16) - self.persistent_frame.astype(np.int16))
        # Shape: (height, width, 3), dtype: int16

        # Identify channels where the difference is <= threshold (noise)
        is_noise = diff <= threshold  # Shape: (height, width, 3), dtype: bool

        # Extract the current channel-specific changes from the mask
        r_changed = (mask & 4) > 0  # Bit 2: R channel
        g_changed = (mask & 2) > 0  # Bit 1: G channel
        b_changed = (mask & 1) > 0  # Bit 0: B channel

        # Clear the bits for channels where the change is considered noise
        r_changed = r_changed & ~is_noise[:, :, 0]
        g_changed = g_changed & ~is_noise[:, :, 1]
        b_changed = b_changed & ~is_noise[:, :, 2]

        # Recompute the mask with the filtered changes
        updated_mask = (r_changed.astype(np.uint8) * 4 +
                        g_changed.astype(np.uint8) * 2 +
                        b_changed.astype(np.uint8) * 1)

        return updated_mask, new_values
