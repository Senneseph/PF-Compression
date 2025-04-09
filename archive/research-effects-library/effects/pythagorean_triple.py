"""
Pythagorean Triple Decoder implementation.
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

class PythagoreanTripleDecoder(Decoder):
    """
    Decoder that reshapes and upscales Pythagorean triple data back into a frame.
    """
    
    def __init__(self, output_shape=(480, 640, 3)):
        """
        Initialize the Pythagorean Triple Decoder.
        
        Args:
            output_shape: Shape of the output frame (height, width, channels).
        """
        self.output_shape = output_shape
    
    def decode(self, encoded_data):
        """
        Decode Pythagorean triple data back into a frame.
        
        Args:
            encoded_data: Tuple (closest_triples, small_shape, output_shape)
                - closest_triples: NumPy array of shape (small_height*small_width, 3) with uint8 values.
                - small_shape: Tuple (small_height, small_width) of the downsampled frame.
                - output_shape: Tuple (height, width, channels) of the output frame.
            
        Returns:
            frame: NumPy array of shape output_shape with uint8 values (BGR or RGB).
        """
        closest_triples, small_shape, output_shape = encoded_data
        small_height, small_width = small_shape
        
        # Reshape the closest triples back into a frame
        small_output = closest_triples.reshape(small_height, small_width, 3)
        
        # Upscale the frame to the original resolution
        output = cv2.resize(small_output, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return self.validate_output_frame(output, output_shape)
