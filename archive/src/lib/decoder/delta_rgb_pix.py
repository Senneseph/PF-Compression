"""
Delta RGB Pixel Decoder implementation.
"""
import numpy as np
import cv2
from abc import ABC, abstractmethod

class Decoder(ABC):
    """
    Abstract base class for all decoders.

    A decoder takes encoded data and transforms it back into a frame.
    """

    @abstractmethod
    def decode(self, encoded_data):
        """
        Decode encoded data back into a frame.

        Args:
            encoded_data: Encoded data in a format specific to the decoder implementation.

        Returns:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        pass

    def validate_output_frame(self, frame, expected_shape, expected_dtype=np.uint8):
        """
        Validate that the output frame has the expected shape and data type.

        Args:
            frame: NumPy array to validate.
            expected_shape: Tuple of expected shape (height, width, channels).
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
        if frame.shape != expected_shape:
            raise ValueError(f"Frame must have shape {expected_shape}, got {frame.shape}")

        # Ensure the frame has the expected data type
        if frame.dtype != expected_dtype:
            frame = frame.astype(expected_dtype)

        return frame

class DeltaRGBPixDecoder(Decoder):
    """
    Decoder that applies changed RGB values to a persistent frame.
    """

    def __init__(self, persistent_frame=None, frame_shape=(480, 640, 3)):
        """
        Initialize the Delta RGB Pixel Decoder.

        Args:
            persistent_frame: Initial persistent frame. If None, a black frame is created.
            frame_shape: Shape of the frame (height, width, channels).
        """
        self.frame_shape = frame_shape
        if persistent_frame is None:
            self.persistent_frame = np.zeros(frame_shape, dtype=np.uint8)
        else:
            self.persistent_frame = persistent_frame.copy()

    def decode(self, encoded_data):
        """
        Decode the frame by applying the changed RGB values to the persistent frame.

        Args:
            encoded_data: tuple (mask, new_values)
                - mask: 2D array of 3-bit values (0-7) indicating which channels changed.
                - new_values: 3D array of new RGB values for the changed channels.

        Returns:
            np.ndarray: Updated persistent frame.
        """
        mask, new_values = encoded_data

        # Extract channel-specific masks
        r_changed = (mask & 4) > 0  # Bit 2: R channel changed
        g_changed = (mask & 2) > 0  # Bit 1: G channel changed
        b_changed = (mask & 1) > 0  # Bit 0: B channel changed
        # Each is shape: (height, width), dtype: bool

        # Update the persistent frame with the new values
        if np.any(r_changed):
            self.persistent_frame[r_changed, 0] = new_values[r_changed, 0]
        if np.any(g_changed):
            self.persistent_frame[g_changed, 1] = new_values[g_changed, 1]
        if np.any(b_changed):
            self.persistent_frame[b_changed, 2] = new_values[b_changed, 2]

        # Calculate statistics
        total_pixels = self.frame_shape[0] * self.frame_shape[1]
        changed_pixels = np.count_nonzero(mask)
        changed_channels = np.count_nonzero(r_changed) + np.count_nonzero(g_changed) + np.count_nonzero(b_changed)

        # Create a copy of the persistent frame for annotation
        output_frame = self.persistent_frame.copy()

        # Overlay statistics on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White text
        thickness = 1

        # Add text with statistics
        cv2.putText(output_frame, f"Changed Pixels: {changed_pixels}/{total_pixels} ({changed_pixels/total_pixels:.1%})",
                   (10, 20), font, font_scale, color, thickness)
        cv2.putText(output_frame, f"Changed Channels: {changed_channels}/{total_pixels*3} ({changed_channels/(total_pixels*3):.1%})",
                   (10, 40), font, font_scale, color, thickness)

        return output_frame

    def get_persistent_frame(self):
        """
        Get the current persistent frame.

        Returns:
            np.ndarray: The current persistent frame.
        """
        return self.persistent_frame.copy()

    def set_persistent_frame(self, frame):
        """
        Set the persistent frame.

        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        self.persistent_frame = self.validate_frame(frame, expected_shape=self.frame_shape)
