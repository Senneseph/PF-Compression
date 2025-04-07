"""
Middle Four Bits Effect implementation.
"""
import numpy as np
from lib.effect.base import Effect

class MiddleFourBitsEffect(Effect):
    """
    Effect that preserves only the middle four bits (bits 2-5) of each pixel in an RGB frame,
    setting all other bits (0, 1, 6, 7) to zero.
    """
    
    def __init__(self):
        """
        Initialize the Middle Four Bits Effect.
        """
        super().__init__(name="Middle Four Bits")
    
    def filter(self, frame):
        """
        Filter a frame by preserving only the middle four bits.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            filtered_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        
        # Create a mask for bits 2-5: 00111100 in binary, which is 60 in decimal
        mask = 0b00111100  # Decimal 60, preserves bits 2, 3, 4, 5
        
        # Apply the mask to all pixels in all channels using bitwise AND
        # This zeros out bits 0, 1, 6, and 7, keeping bits 2-5 unchanged
        filtered_frame = frame & mask
        
        return filtered_frame
    
    def transform(self, frame):
        """
        Transform a frame using the Middle Four Bits effect.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        return self.filter(frame)
