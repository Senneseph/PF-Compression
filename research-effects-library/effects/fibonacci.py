"""
Fibonacci Transformer implementation.
"""
import numpy as np
import cv2
from abc import ABC, abstractmethod

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

class FibonacciTransformer(Transformer):
    """
    Transformer that applies a grayscale quantization and delta-based compression.
    
    This transformer:
    1. Converts the frame to grayscale.
    2. Quantizes the grayscale values to 8 levels.
    3. Computes the delta from the previous frame.
    4. Only updates pixels with significant changes.
    """
    
    def __init__(self, scale=32, threshold=10):
        """
        Initialize the Fibonacci Transformer.
        
        Args:
            scale: Quantization scale (default: 32, resulting in 8 levels).
            threshold: Threshold for delta changes (default: 10).
        """
        self.scale = scale
        self.threshold = threshold
        self.last_frame = None
    
    def transform(self, frame):
        """
        Transform a frame using the Fibonacci transformation.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Quantize to 8 levels
        quantized = (gray // self.scale) * self.scale
        
        # Apply delta-based compression if we have a previous frame
        if self.last_frame is not None:
            # Compute the delta between the current and previous frame
            delta = quantized.astype(np.int16) - self.last_frame.astype(np.int16)
            
            # Only keep significant changes
            delta = np.where(np.abs(delta) > self.threshold, delta, 0)
            
            # Update the quantized frame with the delta
            quantized = (self.last_frame + delta).clip(0, 255).astype(np.uint8)
        
        # Store the current frame for the next iteration
        self.last_frame = quantized.copy()
        
        # Convert back to BGR
        return cv2.cvtColor(quantized, cv2.COLOR_GRAY2BGR)
    
    def reset(self):
        """
        Reset the transformer to its initial state.
        """
        self.last_frame = None
