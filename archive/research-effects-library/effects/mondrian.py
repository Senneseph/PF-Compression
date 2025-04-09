"""
Mondrian Effect implementation.
"""
import numpy as np
import cv2
from research_effects_library.core.effect import Effect

# Mondrian palette definition
MONDRIAN_PALETTE = np.array([
    [227, 66, 52],    # Mondrian Red
    [238, 210, 20],   # Mondrian Yellow
    [39, 89, 180],    # Mondrian Blue
    [255, 255, 255],  # White
    [0, 0, 0],        # Black
])

class MondrianEffect(Effect):
    """
    Effect that creates a Mondrian-style painting from the input frame.
    
    This effect:
    1. Detects edges in the frame to identify rectangular regions
    2. Assigns Mondrian-style colors to each region
    3. Adds subtle gradients within each rectangle
    4. Draws black borders around each region
    
    The result resembles the abstract paintings of Piet Mondrian,
    characterized by primary colors, white, and black grid lines.
    """
    
    def __init__(self):
        """
        Initialize the Mondrian Effect.
        """
        super().__init__(name="Mondrian")
        self.description = (
            "Creates a Mondrian-style painting from the input frame, "
            "with rectangular regions of primary colors (red, yellow, blue) "
            "and white, separated by black grid lines. Named after Dutch painter "
            "Piet Mondrian, known for his abstract geometric compositions."
        )
    
    def find_closest_mondrian_color(self, pixel):
        """
        Find the closest Mondrian palette color for a given pixel.
        
        Args:
            pixel: NumPy array of shape (3,) with RGB values.
            
        Returns:
            np.ndarray: The closest Mondrian palette color.
        """
        distances = np.sqrt(np.sum((MONDRIAN_PALETTE - pixel) ** 2, axis=1))
        return MONDRIAN_PALETTE[np.argmin(distances)]
    
    def transform(self, frame):
        """
        Transform a frame using the Mondrian effect.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        height, width = frame.shape[:2]
        
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to enhance divisions clearly
        edges_dilated = cv2.dilate(edges, np.ones((3, 3)), iterations=2)
        
        # Find contours to identify rectangular regions
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Prepare output frame
        mondrian_frame = np.full_like(frame, 255)  # Start with white background
        
        # Process each region
        for cnt in contours:
            # Approximate contour to rectangles
            x, y, rect_w, rect_h = cv2.boundingRect(cnt)
            
            # Ignore very small regions (noise)
            if rect_w < 20 or rect_h < 20:
                continue
            
            # Extract region of interest
            roi = frame[y:y+rect_h, x:x+rect_w]
            
            # Calculate the average color of region
            avg_color = np.mean(roi.reshape(-1, 3), axis=0).astype(np.uint8)
            
            # Snap average color to closest Mondrian palette color
            mondrian_region_color = self.find_closest_mondrian_color(avg_color)
            
            # Create subtle gradient within the rectangle
            grad_rect = np.zeros((rect_h, rect_w, 3), dtype=np.uint8)
            for i in range(rect_h):
                gradient_factor = 1 - (i / rect_h) * 0.2  # Slight vertical gradient
                grad_rect[i] = np.clip(mondrian_region_color * gradient_factor, 0, 255)
            
            # Apply colored rectangle with gradient
            mondrian_frame[y:y+rect_h, x:x+rect_w] = grad_rect
            
            # Draw black borders (bold outlines)
            cv2.rectangle(mondrian_frame, (x, y), (x+rect_w, y+rect_h), (0, 0, 0), thickness=3)
        
        return mondrian_frame
