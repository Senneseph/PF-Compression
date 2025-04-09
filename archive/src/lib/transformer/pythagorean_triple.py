"""
Pythagorean Triple Transformer implementation.
"""
import numpy as np
import cv2
from abc import ABC, abstractmethod
from src.lib.encoder.pythagorean_triple import PythagoreanTripleEncoder
from src.lib.decoder.pythagorean_triple import PythagoreanTripleDecoder

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

class PythagoreanTripleTransformer(Transformer):
    """
    Transformer that applies the Pythagorean Triple transformation to a frame.
    
    This transformer:
    1. Downsamples the frame for performance.
    2. Encodes the frame by mapping each pixel to the closest Pythagorean triple.
    3. Decodes the encoded data back into a frame and upscales it.
    """
    
    def __init__(self, downsample_factor=2, max_value=255):
        """
        Initialize the Pythagorean Triple Transformer.
        
        Args:
            downsample_factor: Factor by which to downsample the frame for performance.
            max_value: Maximum value for Pythagorean triple components.
        """
        self.downsample_factor = downsample_factor
        self.max_value = max_value
        
        # Initialize the encoder and decoder
        self.encoder = PythagoreanTripleEncoder(max_value)
        self.decoder = PythagoreanTripleDecoder()
    
    def transform(self, frame):
        """
        Transform a frame using the Pythagorean Triple transformation.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        height, width, _ = frame.shape
        
        # Downsample the frame for performance
        small_width = width // self.downsample_factor
        small_height = height // self.downsample_factor
        small_frame = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        
        # Encode the downsampled frame
        closest_triples = self.encoder.encode(small_frame)
        
        # Decode the encoded data
        small_shape = (small_height, small_width)
        output_shape = (height, width, 3)
        output_frame = self.decoder.decode((closest_triples, small_shape, output_shape))
        
        return output_frame
