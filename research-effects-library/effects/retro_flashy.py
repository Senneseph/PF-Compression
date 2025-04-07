"""
Retro Flashy Effect implementation.
"""
import numpy as np
import cv2
from lib.effect.base import Effect

class RetroFlashyEffect(Effect):
    """
    Effect that creates an extremely low-resolution retro look with flashy colors.
    
    This effect:
    1. Drastically reduces resolution to 28x24
    2. Quantizes colors to 8 colors (3-bit color)
    3. Uses delta-based compression with keyframes
    4. Upscales back to original size with nearest-neighbor interpolation
    """
    
    def __init__(self, keyframe_interval=30):
        """
        Initialize the Retro Flashy Effect.
        
        Args:
            keyframe_interval: Number of frames between keyframes (default: 30).
        """
        super().__init__(name="Retro Flashy")
        self.keyframe_interval = keyframe_interval
        self.frame_count = 0
        self.last_frame = None
    
    def transform(self, frame):
        """
        Transform a frame using the Retro Flashy effect.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        height, width = frame.shape[:2]
        
        # Resize to extremely low resolution
        frame_small = cv2.resize(frame, (28, 24))
        
        # Quantize to 8 colors (3-bit LUT)
        frame_rgb = (frame_small // 32) * 32
        
        # Increment frame counter
        self.frame_count += 1
        
        # Delta Frame (every Nth as keyframe)
        if self.last_frame is not None and self.frame_count % self.keyframe_interval != 0:
            delta = frame_rgb.astype(np.int16) - self.last_frame.astype(np.int16)
            delta = np.where(np.abs(delta) > 16, delta // 16, 0)  # ~1 bit/pixel
            frame_rgb = (self.last_frame + delta * 16).clip(0, 255).astype(np.uint8)
        
        # Store current frame for next iteration
        self.last_frame = frame_rgb.copy()
        
        # Upscale for display
        output = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, "RETRO MODE", (20, 30), font, 0.7, (255, 255, 255), 1)
        
        # Add scan line effect
        for y in range(0, height, 2):
            output[y:y+1, :] = (output[y:y+1, :] * 0.7).astype(np.uint8)
        
        return output
    
    def reset(self):
        """
        Reset the effect to its initial state.
        """
        self.frame_count = 0
        self.last_frame = None
