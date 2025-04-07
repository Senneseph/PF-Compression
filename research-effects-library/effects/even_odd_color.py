"""
Even/Odd Color Decoder implementation.
"""
import numpy as np
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

class EvenOddColorDecoder(Decoder):
    """
    Decoder that reverses the even/odd transformation and updates a persistent frame.
    """
    
    def __init__(self, frame_shape=(480, 640, 3)):
        """
        Initialize the Even/Odd Color Decoder.
        
        Args:
            frame_shape: Shape of the frame (height, width, channels).
        """
        self.frame_shape = frame_shape
        self.persistent_frame = np.zeros(frame_shape, dtype=np.uint8)
        self.odd_cycle = True
    
    def decode(self, encoded_data):
        """
        Decode the frame by reversing the even/odd transformation and updating the persistent frame.
        
        Args:
            encoded_data: Tuple (transformed_frame, changed_mask, is_odd)
                - transformed_frame: NumPy array of shape (height, width, 3) with uint8 values.
                - changed_mask: Boolean mask indicating which pixels were changed.
                - is_odd: Boolean indicating whether odd values were enforced.
            
        Returns:
            np.ndarray: Updated persistent frame.
        """
        transformed_frame, changed_mask, is_odd = encoded_data
        
        # Read the current parity mode
        odd = is_odd
        
        # Determine the target parity and adjustment
        target_parity = 1 if odd else 0
        adjustment = -1 if odd else 1  # Subtract 1 if odd=True, add 1 if odd=False
        boundary_check = (transformed_frame > 0) if odd else (transformed_frame < 255)
        
        # Reverse the transformation
        condition = changed_mask & (transformed_frame % 2 == target_parity) & boundary_check
        decoded_frame = np.where(condition, transformed_frame + adjustment, transformed_frame)
        
        # Update the persistent frame
        self.persistent_frame = np.where(changed_mask[:, :, np.newaxis], decoded_frame, self.persistent_frame).astype(np.uint8)
        
        # Update the odd cycle for the next frame
        self.odd_cycle = not odd
        
        return self.persistent_frame
    
    def reset(self):
        """
        Reset the persistent frame to all zeros.
        """
        self.persistent_frame = np.zeros(self.frame_shape, dtype=np.uint8)
