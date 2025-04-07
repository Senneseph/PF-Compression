"""
1970s Panavision Filter implementation.
"""
import numpy as np
import cv2
from abc import ABC, abstractmethod

class Filter(ABC):
    """
    Abstract base class for all filters.
    
    A filter takes an input frame, applies a filtering operation, and returns a filtered frame.
    """
    
    @abstractmethod
    def filter(self, frame):
        """
        Filter a frame.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            filtered_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
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

class Panavision1970sFilter(Filter):
    """
    Filter that emulates 1970s Panavision color grading, like Conan the Barbarian.
    
    This filter:
    1. Adjusts color temperature and tint to mimic Kodak 5247 film stock.
    2. Increases contrast and crushes blacks.
    3. Applies lens softness and bloom effects.
    4. Adds film grain.
    """
    
    def __init__(self, grain_intensity=10, bloom_threshold=200):
        """
        Initialize the 1970s Panavision Filter.
        
        Args:
            grain_intensity: Intensity of the film grain effect (default: 10).
            bloom_threshold: Threshold for bloom effect (default: 200).
        """
        self.grain_intensity = grain_intensity
        self.bloom_threshold = bloom_threshold
    
    def filter(self, frame):
        """
        Filter a frame with 1970s Panavision aesthetic.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            filtered_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        
        # Step 1: Color Adjustments (Mimicking Kodak 5247 and 1970s grading)
        # Convert to float for processing
        frame_float = frame.astype(np.float32) / 255.0
        
        # Adjust temperature (warm shift) and tint
        frame_float[:, :, 1] *= 0.95  # Normalize green channel
        frame_float[:, :, 2] *= 0.9975  # Boost red channel (wrt green)
        frame_float[:, :, 0] *= 0.9025  # Reduce blue channel (wrt green)
        
        # Increase contrast and crush blacks
        frame_float = (frame_float - 0.1) * 1.5  # Shift and scale for higher contrast
        frame_float = np.clip(frame_float, 0, 1)
        
        # Step 2: Apply LUT (Placeholder)
        frame_adjusted = (frame_float * 255).astype(np.uint8)
        
        # Step 3: Emulate Lens Softness
        frame_soft = cv2.GaussianBlur(frame_adjusted, (5, 5), sigmaX=0.5)
        
        # Step 4: Bloom Effect for Highlights
        gray = cv2.cvtColor(frame_adjusted, cv2.COLOR_BGR2GRAY)
        highlights = gray > self.bloom_threshold
        bloom = cv2.GaussianBlur(frame_adjusted.astype(np.float32), (15, 15), sigmaX=2.0)
        bloom = bloom.astype(np.uint8)
        frame_soft[highlights] = 0.7 * frame_soft[highlights] + 0.3 * bloom[highlights]
        
        # Step 5: Add Film Grain
        grain = np.random.normal(0, self.grain_intensity, frame_soft.shape).astype(np.float32)
        grain[:, :, 0] *= 1.2  # More grain in blue channel
        frame_with_grain = frame_soft.astype(np.float32) + grain
        frame_with_grain = np.clip(frame_with_grain, 0, 255).astype(np.uint8)
        
        return frame_with_grain
