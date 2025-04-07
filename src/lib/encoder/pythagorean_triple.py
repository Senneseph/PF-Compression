"""
Pythagorean Triple Encoder implementation.
"""
import numpy as np
from abc import ABC, abstractmethod
from src.utils import generate_pythagorean_triples

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

class PythagoreanTripleEncoder(Encoder):
    """
    Encoder that maps each pixel to the closest Pythagorean triple.
    """
    
    def __init__(self, max_value=255):
        """
        Initialize the Pythagorean Triple Encoder.
        
        Args:
            max_value: Maximum value for Pythagorean triple components.
        """
        self.max_value = max_value
        self.pythagorean_triples = np.array(generate_pythagorean_triples(max_value), dtype=np.uint8)
    
    def encode(self, frame):
        """
        Encode a frame by mapping each pixel to the closest Pythagorean triple.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            closest_triples: NumPy array of shape (height*width, 3) with uint8 values.
        """
        frame = self.validate_frame(frame)
        
        # Reshape the frame to a 2D array of pixels
        pixels = frame.reshape(-1, 3)
        
        # Calculate the absolute difference between each pixel and each triple
        diffs = np.abs(pixels[:, np.newaxis, :] - self.pythagorean_triples[np.newaxis, :, :])
        
        # Sum the differences across the RGB channels
        distances = diffs.sum(axis=2)
        
        # Find the index of the closest triple for each pixel
        closest_indices = np.argmin(distances, axis=1)
        
        # Get the closest triples
        closest_triples = self.pythagorean_triples[closest_indices]
        
        return closest_triples
