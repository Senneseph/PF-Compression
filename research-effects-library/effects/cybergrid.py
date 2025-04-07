"""
Cybergrid Effect implementation.
"""
import numpy as np
import cv2
from lib.effect.base import Effect

class CybergridEffect(Effect):
    """
    Effect that creates a cyberpunk-style grid overlay with vectorized edges.
    
    This effect:
    1. Detects edges in the frame
    2. Vectorizes the edges into simplified polygons
    3. Draws a grid overlay
    4. Adds neon glow effects
    """
    
    def __init__(self, max_shapes=512, grid_spacing=20):
        """
        Initialize the Cybergrid Effect.
        
        Args:
            max_shapes: Maximum number of shapes to detect and draw (default: 512).
            grid_spacing: Spacing between grid lines in pixels (default: 20).
        """
        super().__init__(name="Cybergrid")
        self.max_shapes = max_shapes
        self.grid_spacing = grid_spacing
    
    def transform(self, frame):
        """
        Transform a frame using the Cybergrid effect.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find and limit contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.max_shapes]
        
        # Create output canvas
        height, width = frame.shape[:2]
        output = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw vectorized edges in neon cyan
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(output, [approx], -1, (0, 255, 255), 1)
        
        # Draw grid overlay
        for x in range(0, width, self.grid_spacing):
            cv2.line(output, (x, 0), (x, height), (0, 50, 50), 1)
        
        for y in range(0, height, self.grid_spacing):
            cv2.line(output, (0, y), (width, y), (0, 50, 50), 1)
        
        # Add glow effect
        glow = cv2.GaussianBlur(output, (5, 5), 0)
        output = cv2.addWeighted(output, 1.0, glow, 0.5, 0)
        
        # Blend with original frame
        alpha = 0.7
        output = cv2.addWeighted(frame, 1 - alpha, output, alpha, 0)
        
        # Add HUD elements
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, "CYBERGRID v1.0", (20, 30), font, 0.7, (0, 255, 255), 1)
        cv2.putText(output, f"OBJECTS: {len(contours)}", (20, 60), font, 0.5, (0, 255, 255), 1)
        
        # Add scan line effect
        scan_line_y = int((height * (np.sin(cv2.getTickCount() / 5000000.0) + 1) / 2))
        cv2.line(output, (0, scan_line_y), (width, scan_line_y), (0, 255, 255), 1)
        
        return output
