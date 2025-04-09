"""
Abstract base class for decoders.
"""
from abc import ABC, abstractmethod
import numpy as np

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
