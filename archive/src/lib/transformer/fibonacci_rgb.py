"""
Fibonacci RGB Transformer implementation.
"""
import numpy as np
from abc import ABC, abstractmethod
from src.utils import precompute_nearest_fibonacci

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

class FibonacciRGBTransformer(Transformer):
    """
    Transformer that maps each RGB value to the nearest Fibonacci number.
    """
    
    def __init__(self, max_value=255):
        """
        Initialize the Fibonacci RGB Transformer.
        
        Args:
            max_value: Maximum value for the lookup table.
        """
        self.max_value = max_value
        self.fib_lookup = precompute_nearest_fibonacci(max_value)
    
    def transform(self, frame):
        """
        Transform a frame by mapping each RGB value to the nearest Fibonacci number.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        
        # Apply the Fibonacci lookup table to the frame
        output = self.fib_lookup[frame]
        
        return output
