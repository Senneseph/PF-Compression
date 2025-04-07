"""
Interlaced Horizontal Effect implementation.
"""
import numpy as np
import cv2
from research_effects_library.core.effect import Effect

class InterlacedHorizontalEffect(Effect):
    """
    Effect that applies horizontal interlacing (row-based).
    
    This effect:
    1. Alternates between transmitting even and odd rows each frame
    2. Updates only half the rows in each frame
    3. Maintains a persistent buffer for progressive reconstruction
    """
    
    def __init__(self, frame_shape=(480, 640, 3)):
        """
        Initialize the Interlaced Horizontal Effect.
        
        Args:
            frame_shape: Shape of the frame (height, width, channels).
        """
        super().__init__(name="Interlaced Horizontal")
        self.frame_shape = frame_shape
        self.frame_count = 0
        self.last_frame = np.zeros(frame_shape, dtype=np.uint8)
    
    def encode(self, frame):
        """
        Encode a frame for horizontal interlacing by selecting even or odd rows.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            tuple: (rows_data, is_even_frame, height)
                - rows_data: NumPy array of the selected rows.
                - is_even_frame: bool, True if even rows are selected.
                - height: int, frame height.
        """
        frame = self.validate_frame(frame, expected_shape=self.frame_shape)
        
        self.frame_count += 1
        is_even_frame = (self.frame_count % 2 == 0)
        height = frame.shape[0]
        
        if is_even_frame:
            rows_data = frame[0:height:2, :, :].copy()  # Even rows (0, 2, 4, ...)
        else:
            rows_data = frame[1:height:2, :, :].copy()  # Odd rows (1, 3, 5, ...)
        
        return rows_data, is_even_frame, height
    
    def decode(self, encoded_data):
        """
        Decode the interlaced data by updating even or odd rows in the persistent frame.
        
        Args:
            encoded_data: tuple (rows_data, is_even_frame, height)
                - rows_data: NumPy array of the selected rows.
                - is_even_frame: bool, True if even rows are selected.
                - height: int, frame height.
            
        Returns:
            np.ndarray: Updated frame with shape self.frame_shape.
        """
        rows_data, is_even_frame, height = encoded_data
        
        output_frame = self.last_frame.copy()
        
        if is_even_frame:
            output_frame[0:height:2, :, :] = rows_data
        else:
            output_frame[1:height:2, :, :] = rows_data
        
        self.last_frame = output_frame.copy()
        
        # Add visual indicator of which rows were updated
        width = output_frame.shape[1]
        indicator_width = 20
        indicator = np.zeros((height, indicator_width, 3), dtype=np.uint8)
        
        if is_even_frame:
            indicator[0:height:2, :, 1] = 255  # Green for even rows
        else:
            indicator[1:height:2, :, 0] = 255  # Blue for odd rows
        
        output_frame[:, :indicator_width, :] = indicator
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Even Rows" if is_even_frame else "Odd Rows"
        cv2.putText(output_frame, text, (indicator_width + 10, 20), font, 0.5, (255, 255, 255), 1)
        
        # Compute analytics
        rows_updated = rows_data.shape[0]  # Number of rows updated
        data_size_bits = rows_updated * width * 3 * 8  # Rows × cols × channels × bits
        frame_rate = 30
        data_rate_kbps = (data_size_bits * frame_rate) // 1000
        
        analytics_text = f"Updated: {rows_updated} rows, Rate: {data_rate_kbps} kbps"
        cv2.putText(output_frame, analytics_text, (indicator_width + 10, 40), font, 0.5, (255, 255, 255), 1)
        
        return output_frame
    
    def transform(self, frame):
        """
        Transform a frame using horizontal interlacing.
        
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
