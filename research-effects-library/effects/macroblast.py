"""
Macroblast Effect implementation.
"""
import numpy as np
import cv2
from lib.effect.base import Effect

class MacroblastEffect(Effect):
    """
    Effect that creates an exaggerated JPEG compression artifact look.
    
    This effect:
    1. Downscales the frame to exaggerate blockiness
    2. Applies JPEG compression at low quality
    3. Upscales back to original size with nearest-neighbor interpolation
    4. Enhances block edges with a cyan overlay
    """
    
    def __init__(self, block_size=8, quality=5):
        """
        Initialize the Macroblast Effect.
        
        Args:
            block_size: Size of the macroblocks in pixels (default: 8).
            quality: JPEG quality level (1-100, lower = more artifacts) (default: 5).
        """
        super().__init__(name="Macroblast")
        self.block_size = block_size
        self.quality = quality
    
    def transform(self, frame):
        """
        Transform a frame using the Macroblast effect.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        height, width = frame.shape[:2]
        
        # Downscale frame to exaggerate blockiness (e.g., 1/4 original size)
        small_width = width // 4
        small_height = height // 4
        small_frame = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_NEAREST)
        
        # Encode and decode with JPEG at low quality to introduce macroblocking
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encoded = cv2.imencode('.jpg', small_frame, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Upscale back to original size with nearest-neighbor to keep blocks sharp
        blocky_frame = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Enhance block edges for extra "JPEG-y" effect
        gray = cv2.cvtColor(blocky_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Convert edges to BGR and apply cyan mask directly
        edges_colored = np.zeros_like(blocky_frame, dtype=np.uint8)
        edges_colored[edges != 0] = [0, 255, 255]  # Cyan where edges exist
        
        # Blend with explicit type handling
        blocky_frame = cv2.addWeighted(blocky_frame, 0.8, edges_colored, 0.2, 0, dtype=cv2.CV_8U)
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(blocky_frame, f"JPEG Q:{self.quality}", (20, 30), font, 0.7, (0, 255, 255), 1)
        
        return blocky_frame
