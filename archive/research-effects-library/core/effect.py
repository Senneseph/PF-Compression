"""
Abstract base class for all effects.
"""
import numpy as np
from abc import ABC, abstractmethod

class Effect(ABC):
    """
    Abstract base class for all effects.
    
    An effect combines encoding, decoding, filtering, and transformation operations
    into a single cohesive unit. It provides a standard interface for all effects
    in the library.
    """
    
    def __init__(self, name=None):
        """
        Initialize the effect.
        
        Args:
            name: Optional name for the effect.
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def transform(self, frame):
        """
        Transform a frame using this effect.
        
        This is the main method that should be called to apply the effect.
        It typically combines encoding, decoding, and filtering operations.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        pass
    
    def encode(self, frame):
        """
        Encode a frame into a different representation.
        
        This method should be overridden by effects that need to encode frames.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            encoded_data: Encoded data in a format specific to the effect.
        """
        return frame
    
    def decode(self, encoded_data):
        """
        Decode encoded data back into a frame.
        
        This method should be overridden by effects that need to decode frames.
        
        Args:
            encoded_data: Encoded data in a format specific to the effect.
            
        Returns:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        return encoded_data
    
    def filter(self, frame):
        """
        Apply a filtering operation to a frame.
        
        This method should be overridden by effects that need to filter frames.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            filtered_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        return frame
    
    def reset(self):
        """
        Reset the effect to its initial state.
        
        This method should be overridden by effects that maintain state.
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
