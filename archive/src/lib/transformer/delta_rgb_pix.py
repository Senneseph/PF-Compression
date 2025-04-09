"""
Delta RGB Pixel Transformer implementation.
"""
import numpy as np
from abc import ABC, abstractmethod
from src.lib.encoder.delta_rgb_pix import DeltaRGBPixEncoder
from src.lib.decoder.delta_rgb_pix import DeltaRGBPixDecoder

class Transformer(ABC):
    """
    Abstract base class for all transformers.

    A transformer takes an input frame, applies a transformation, and returns a transformed frame.
    It may use an encoder and decoder internally, or apply the transformation directly.
    """

    @abstractmethod
    def transform(self, frame):
        """
        Transform a frame.

        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).

        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
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

class DeltaRGBPixTransformer(Transformer):
    """
    Transformer that applies the Delta RGB Pixel transformation to a frame.

    This transformer:
    1. Encodes the frame by identifying changed RGB values compared to a persistent frame.
    2. Optionally denoises the encoded data to remove small changes.
    3. Decodes the encoded data back into a frame.
    """

    def __init__(self, frame_shape=(480, 640, 3), denoise_threshold=15):
        """
        Initialize the Delta RGB Pixel Transformer.

        Args:
            frame_shape: Shape of the frame (height, width, channels).
            denoise_threshold: Threshold for denoising (default: 15).
        """
        self.frame_shape = frame_shape
        self.denoise_threshold = denoise_threshold

        # Create a shared persistent frame for the encoder and decoder
        self.persistent_frame = np.zeros(frame_shape, dtype=np.uint8)

        # Initialize the encoder and decoder
        self.encoder = DeltaRGBPixEncoder(self.persistent_frame, frame_shape)
        self.decoder = DeltaRGBPixDecoder(self.persistent_frame, frame_shape)

    def transform(self, frame):
        """
        Transform a frame using the Delta RGB Pixel transformation.

        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).

        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame, expected_shape=self.frame_shape)

        # Encode the frame
        mask, new_values = self.encoder.encode(frame)

        # Denoise the encoded data
        mask, new_values = self.encoder.denoise(mask, new_values, frame, self.denoise_threshold)

        # Decode the encoded data
        output_frame = self.decoder.decode((mask, new_values))

        return output_frame

    def reset(self):
        """
        Reset the persistent frame to all zeros.
        """
        self.persistent_frame = np.zeros(self.frame_shape, dtype=np.uint8)
        self.encoder.persistent_frame = self.persistent_frame
        self.decoder.persistent_frame = self.persistent_frame
