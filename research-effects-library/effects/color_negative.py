"""
Color Negative Effect implementation.
"""
import numpy as np
from lib.effect.base import Effect

class ColorNegativeEffect(Effect):
    """
    Effect that converts an RGB frame into its color negative by inverting each channel.
    For each pixel value in each channel, the new value is 255 - original_value.
    """
    
    def __init__(self):
        """
        Initialize the Color Negative Effect.
        """
        super().__init__(name="Color Negative")
    
    def filter(self, frame):
        """
        Filter a frame by converting it to its color negative.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            filtered_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        
        # Invert the frame: new_value = 255 - original_value for each channel
        negative_frame = 255 - frame
        
        return negative_frame
    
    def transform(self, frame):
        """
        Transform a frame using the Color Negative effect.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        return self.filter(frame)
