"""
Matrix Digital Rain Effect implementation.
"""
import numpy as np
import cv2
import random
from research_effects_library.core.effect import Effect

class MatrixColumn:
    """
    A column of falling characters in the Matrix Digital Rain effect.
    """
    
    def __init__(self, x, height, speed):
        """
        Initialize a Matrix column.
        
        Args:
            x: X-coordinate of the column.
            height: Height of the frame.
            speed: Speed of falling (pixels per frame).
        """
        self.x = x  # X-coordinate of the column
        self.height = height  # Frame height
        self.speed = speed  # Speed of falling (pixels per frame)
        self.head_pos = random.randint(0, height)  # Starting position of the head
        self.characters = []  # List of (y_position, character) tuples
        # Katakana characters (subset for simplicity) + numbers
        self.char_set = [chr(c) for c in range(0x30A2, 0x30FF, 2)] + [str(i) for i in range(10)]
    
    def update(self):
        """
        Update the column by moving the head and adding/removing characters.
        """
        # Move the head downward
        self.head_pos += self.speed
        if self.head_pos > self.height:
            self.head_pos = 0  # Reset to top when it reaches the bottom
        
        # Add a new character at the head position
        new_char = random.choice(self.char_set)
        self.characters.append((self.head_pos, new_char))
        
        # Remove characters that are too far from the head (more than 400 pixels behind)
        self.characters = [(y, char) for y, char in self.characters if self.head_pos - y <= 400]
    
    def draw(self, frame, bright_green, dark_green, font_scale):
        """
        Draw the column on the frame.
        
        Args:
            frame: Frame to draw on.
            bright_green: Bright green color for the head character.
            dark_green: Dark green color for trailing characters.
            font_scale: Font scale for the characters.
        """
        for i, (y, char) in enumerate(self.characters):
            # Calculate distance from head to determine brightness
            distance = self.head_pos - y
            if distance < 0:
                continue  # Skip characters below the head after wrapping
            
            # Interpolate color from bright green to dark green
            fade_factor = min(distance / 200, 1.0)  # Fade over 200 pixels
            color = tuple(int(bright * (1 - fade_factor) + dark * fade_factor) 
                         for bright, dark in zip(bright_green, dark_green))
            
            # Draw the character
            y_pos = int(y) % self.height
            cv2.putText(frame, char, (self.x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 1, cv2.LINE_AA)

class MatrixEffect(Effect):
    """
    Effect that creates a Matrix Digital Rain overlay on the input frame.
    
    This effect:
    1. Creates columns of falling characters (primarily Katakana and numbers)
    2. Renders the characters with a bright green head and fading trail
    3. Blends the character overlay with the original frame
    
    The result resembles the iconic "digital rain" from the Matrix movie series.
    """
    
    def __init__(self, column_width=20, alpha=0.9):
        """
        Initialize the Matrix Effect.
        
        Args:
            column_width: Width between columns in pixels (default: 20).
            alpha: Transparency of the Matrix effect (0 = transparent, 1 = opaque) (default: 0.9).
        """
        super().__init__(name="Matrix Digital Rain")
        self.column_width = column_width
        self.alpha = alpha
        self.columns = None
        self.font_scale = 0.6
        self.description = (
            "Creates a Matrix Digital Rain effect overlay on the input frame, "
            "with falling Katakana characters and numbers in bright green. "
            "The effect resembles the iconic 'digital rain' from the Matrix movie series."
        )
    
    def transform(self, frame):
        """
        Transform a frame using the Matrix Digital Rain effect.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        height, width = frame.shape[:2]
        
        # Define colors
        bright_green = (0, 255, 0)  # Bright green for the head character
        dark_green = (0, 60, 0)     # Dark green for trailing characters
        
        # Create a black overlay for the Matrix effect
        overlay = np.zeros_like(frame)
        
        # Initialize columns if not already done
        if self.columns is None:
            num_columns = width // self.column_width
            self.columns = [
                MatrixColumn(x * self.column_width, height, random.randint(5, 15))
                for x in range(num_columns)
            ]
        
        # Update and draw each column
        for column in self.columns:
            column.update()
            column.draw(overlay, bright_green, dark_green, self.font_scale)
        
        # Blend the overlay with the original frame
        output = cv2.addWeighted(frame, 1 - self.alpha, overlay, self.alpha, 0.0)
        
        return output
    
    def reset(self):
        """
        Reset the effect to its initial state.
        """
        self.columns = None
