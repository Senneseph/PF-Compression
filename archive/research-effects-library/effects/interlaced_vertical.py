"""
Interlaced Vertical Effect implementation.
"""
import numpy as np
import cv2
from research_effects_library.core.effect import Effect

class InterlacedVerticalEffect(Effect):
    """
    Effect that applies vertical interlacing (column-based).
    
    This effect:
    1. Alternates between transmitting even and odd columns each frame
    2. Updates only half the columns in each frame
    3. Maintains a persistent buffer for progressive reconstruction
    """
    
    def __init__(self, frame_shape=(480, 640, 3)):
        """
        Initialize the Interlaced Vertical Effect.
        
        Args:
            frame_shape: Shape of the frame (height, width, channels).
        """
        super().__init__(name="Interlaced Vertical")
        self.frame_shape = frame_shape
        self.frame_count = 0
        self.last_frame = np.zeros(frame_shape, dtype=np.uint8)
    
    def encode(self, frame):
        """
        Encode a frame for vertical interlacing by selecting even or odd columns.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            tuple: (cols_data, is_even_frame, width)
                - cols_data: NumPy array of the selected columns.
                - is_even_frame: bool, True if even columns are selected.
                - width: int, frame width.
        """
        frame = self.validate_frame(frame, expected_shape=self.frame_shape)
        
        self.frame_count += 1
        is_even_frame = (self.frame_count % 2 == 0)
        width = frame.shape[1]
        
        if is_even_frame:
            cols_data = frame[:, 0:width:2, :].copy()  # Even columns (0, 2, 4, ...)
        else:
            cols_data = frame[:, 1:width:2, :].copy()  # Odd columns (1, 3, 5, ...)
        
        return cols_data, is_even_frame, width
    
    def decode(self, encoded_data):
        """
        Decode the interlaced data by updating even or odd columns in the persistent frame.
        
        Args:
            encoded_data: tuple (cols_data, is_even_frame, width)
                - cols_data: NumPy array of the selected columns.
                - is_even_frame: bool, True if even columns are selected.
                - width: int, frame width.
            
        Returns:
            np.ndarray: Updated frame with shape self.frame_shape.
        """
        cols_data, is_even_frame, width = encoded_data
        
        output_frame = self.last_frame.copy()
        
        if is_even_frame:
            output_frame[:, 0:width:2, :] = cols_data
        else:
            output_frame[:, 1:width:2, :] = cols_data
        
        self.last_frame = output_frame.copy()
        
        # Add visual indicator of which columns were updated
        height = output_frame.shape[0]
        indicator_height = 20
        indicator = np.zeros((indicator_height, width, 3), dtype=np.uint8)
        
        if is_even_frame:
            indicator[:, 0:width:2, 1] = 255  # Green for even columns
        else:
            indicator[:, 1:width:2, 0] = 255  # Blue for odd columns
        
        output_frame[:indicator_height, :, :] = indicator
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Even Columns" if is_even_frame else "Odd Columns"
        cv2.putText(output_frame, text, (10, indicator_height + 20), font, 0.5, (255, 255, 255), 1)
        
        return output_frame
    
    def transform(self, frame):
        """
        Transform a frame using vertical interlacing.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        # Ensure the frame is in the correct format
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        frame = self.validate_frame(frame, expected_shape=self.frame_shape)
        
        # Encode the frame
        encoded_data = self.encode(frame)
        
        # Decode the encoded data
        output_frame = self.decode(encoded_data)
        
        return output_frame
    
    def reset(self):
        """
        Reset the effect to its initial state.
        """
        self.frame_count = 0
        self.last_frame = np.zeros(self.frame_shape, dtype=np.uint8)
