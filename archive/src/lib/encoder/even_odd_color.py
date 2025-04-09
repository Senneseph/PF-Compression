"""
Even/Odd Color Encoder implementation.
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

class EvenOddColorEncoder(Encoder):
    """
    Encoder that applies an even/odd transformation to pixel values.
    
    This encoder alternates between forcing odd and even values each frame.
    """
    
    def __init__(self, initial_odd=True):
        """
        Initialize the Even/Odd Color Encoder.
        
        Args:
            initial_odd: Whether to start with odd values (True) or even values (False).
        """
        self.odd_cycle = initial_odd
    
    def encode(self, frame):
        """
        Encode a frame by applying the odd/even transformation.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            tuple: (transformed_frame, changed_mask, is_odd)
                - transformed_frame: NumPy array of shape (height, width, 3) with uint8 values.
                - changed_mask: Boolean mask indicating which pixels were changed.
                - is_odd: Boolean indicating whether odd values were enforced.
        """
        frame = self.validate_frame(frame)
        
        # Read the current parity mode
        odd = self.odd_cycle
        
        # Compute the mask of changed pixels
        target_parity = 1 if odd else 0
        boundary_value = 255 if odd else 0
        changed_mask = (frame % 2 != target_parity) & (frame != boundary_value)
        
        # Apply the transformation in one line
        adjustment = 2 * odd - 1  # 1 if odd=True, -1 if odd=False
        transformed_frame = np.where(changed_mask, frame + adjustment, frame)
        
        # Toggle the odd cycle for the next frame
        self.odd_cycle = not self.odd_cycle
        
        return transformed_frame.astype(np.uint8), changed_mask, odd
