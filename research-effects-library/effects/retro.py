"""
Retro Transformer implementation.
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

class RetroTransformer(Transformer):
    """
    Transformer that applies a retro-style effect with low resolution and color quantization.
    
    This transformer:
    1. Downsamples the frame to a lower resolution.
    2. Quantizes the colors to create a retro look.
    3. Uses delta-based compression for temporal coherence.
    4. Upsamples the result back to the original resolution.
    """
    
    def __init__(self, target_size=(320, 240), quantize_level=64, keyframe_interval=30, delta_threshold=32):
        """
        Initialize the Retro Transformer.
        
        Args:
            target_size: Size to downsample to (width, height).
            quantize_level: Color quantization level (higher values = fewer colors).
            keyframe_interval: Number of frames between keyframes.
            delta_threshold: Threshold for delta changes.
        """
        self.target_size = target_size
        self.quantize_level = quantize_level
        self.keyframe_interval = keyframe_interval
        self.delta_threshold = delta_threshold
        self.last_frame = None
        self.frame_count = 0
    
    def transform(self, frame):
        """
        Transform a frame using the Retro transformation.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        original_height, original_width = frame.shape[:2]
        
        # Downsample the frame
        small_frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Quantize the colors
        frame_rgb = (small_frame // self.quantize_level) * self.quantize_level
        
        # Increment the frame counter
        self.frame_count += 1
        
        # Apply delta-based compression if we have a previous frame and it's not a keyframe
        if self.last_frame is not None and self.frame_count % self.keyframe_interval != 0:
            # Compute the delta between the current and previous frame
            delta = frame_rgb.astype(np.int16) - self.last_frame.astype(np.int16)
            
            # Quantize the delta to reduce noise
            delta = np.where(np.abs(delta) > self.delta_threshold, 
                             delta // (self.delta_threshold // 2), 0)
            
            # Update the frame with the quantized delta
            frame_rgb = (self.last_frame + delta * (self.delta_threshold // 2)).clip(0, 255).astype(np.uint8)
        
        # Store the current frame for the next iteration
        self.last_frame = frame_rgb.copy()
        
        # Upsample back to the original resolution
        return cv2.resize(frame_rgb, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    def reset(self):
        """
        Reset the transformer to its initial state.
        """
        self.last_frame = None
        self.frame_count = 0
