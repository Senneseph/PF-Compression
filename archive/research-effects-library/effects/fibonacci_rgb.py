"""
Fibonacci RGB Effect implementation.
"""
import numpy as np
from lib.effect.base import Effect
from lib.utils import precompute_nearest_fibonacci

class FibonacciRGBEffect(Effect):
    """
    Effect that maps each RGB value to the nearest Fibonacci number.
    """
    
    def __init__(self, max_value=255):
        """
        Initialize the Fibonacci RGB Effect.
        
        Args:
            max_value: Maximum value for the lookup table.
        """
        super().__init__(name="Fibonacci RGB")
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
