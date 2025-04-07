"""
RGB Matrix Strobe Effect implementation.
"""
import numpy as np
import cv2
from research_effects_library.core.effect import Effect

class RGBMatrixStrobeEffect(Effect):
    """
    Effect that transmits one RGB channel with a checkerboard pattern per frame.
    
    This effect:
    1. Cycles through the RGB channels (R, G, B) and parity (even, odd)
    2. Extracts the current channel and keeps only pixels in a checkerboard pattern
    3. Updates the persistent frame with the new channel data
    4. Progressively reconstructs the image over 6 frames
    
    This creates a progressive channel-based transmission effect where the image
    quality improves over time as more channel data is transmitted.
    """
    
    def __init__(self):
        """
        Initialize the RGB Matrix Strobe Effect.
        """
        super().__init__(name="RGB Matrix Strobe")
        self.persistent_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cycle = 0
        self.cycle_order = [
            (0, True),   # R odd
            (1, False),  # G even
            (2, True),   # B odd
            (0, False),  # R even
            (1, True),   # G odd
            (2, False)   # B even
        ]
        self.description = (
            "Transmits one RGB channel with a checkerboard pattern per frame, "
            "cycling through all combinations. This creates a progressive channel-based "
            "transmission effect where the image quality improves over time as more "
            "channel data is transmitted in a spatial pattern."
        )
    
    def encode(self, frame):
        """
        Encode the frame by extracting the specified RGB channel and keeping only pixels in a checkerboard pattern.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            tuple: (output_frame, channel_idx, isOdd)
                - output_frame: NumPy array with only the selected channel and checkerboard pattern.
                - channel_idx: Integer (0=R, 1=G, 2=B) indicating the channel.
                - isOdd: Boolean indicating whether to keep odd (True) or even (False) pixels.
        """
        frame = self.validate_frame(frame)
        
        # Get the current channel index and parity
        channel_idx, isOdd = self.cycle_order[self.cycle]
        
        # Extract just the selected channel
        channel_data = frame[:, :, channel_idx].copy()
        
        # Create a checkerboard mask for even/odd pixels
        height, width = channel_data.shape
        row_indices, col_indices = np.indices((height, width))
        checkerboard = (row_indices + col_indices) % 2  # 0 for even, 1 for odd
        update_mask = (checkerboard == (1 if isOdd else 0))  # True for pixels to update
        
        # Create an output frame with all channels set to 0
        output_frame = np.zeros_like(frame, dtype=np.uint8)
        
        # Apply the update mask to keep only matching values
        output_frame[:, :, channel_idx] = np.where(update_mask, channel_data, 0)
        
        return output_frame, channel_idx, isOdd
    
    def decode(self, encoded_data):
        """
        Decode the encoded data by updating the persistent frame with the transmitted channel data.
        
        Args:
            encoded_data: tuple (output_frame, channel_idx, isOdd)
                - output_frame: NumPy array with only the selected channel and checkerboard pattern.
                - channel_idx: Integer (0=R, 1=G, 2=B) indicating the channel.
                - isOdd: Boolean indicating whether odd (True) or even (False) pixels were kept.
            
        Returns:
            np.ndarray: Updated persistent frame.
        """
        encoded_frame, channel_idx, isOdd = encoded_data
        
        # Extract the transmitted channel data
        new_data = encoded_frame[:, :, channel_idx]
        
        # Create the same checkerboard mask as in the encode function
        height, width = new_data.shape
        row_indices, col_indices = np.indices((height, width))
        checkerboard = (row_indices + col_indices) % 2
        update_mask = (checkerboard == (1 if isOdd else 0))
        
        # Apply updates only where the update mask is True
        self.persistent_frame[:, :, channel_idx] = np.where(
            update_mask, new_data, self.persistent_frame[:, :, channel_idx]
        )
        
        return self.persistent_frame.copy()
    
    def transform(self, frame):
        """
        Transform a frame using the RGB Matrix Strobe effect.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        
        # Encode the frame
        encoded_data = self.encode(frame)
        
        # Decode the encoded data
        output_frame = self.decode(encoded_data)
        
        # Increment the cycle, wrapping back to 0 after 5
        self.cycle = (self.cycle + 1) % 6
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White text
        thickness = 1
        
        # Add channel and parity information
        channel_names = ["Red", "Green", "Blue"]
        channel_idx = encoded_data[1]
        isOdd = encoded_data[2]
        pattern_text = "Odd" if isOdd else "Even"
        cv2.putText(output_frame, f"Channel: {channel_names[channel_idx]} ({pattern_text})", (10, 20), 
                   font, font_scale, font_color, thickness)
        
        # Add data rate information
        frame_height, frame_width = frame.shape[:2]
        bits_per_frame = frame_height * frame_width // 4  # Quarter of the pixels for one channel
        cv2.putText(output_frame, f"Data Rate: {bits_per_frame} bits/frame", (10, 40), 
                   font, font_scale, font_color, thickness)
        
        # Add progress information
        progress = ((self.cycle) % 6) * 16.7
        cv2.putText(output_frame, f"Progress: {progress:.1f}%", (10, 60), 
                   font, font_scale, font_color, thickness)
        
        # Visualize the checkerboard pattern
        height, width = frame.shape[:2]
        row_indices, col_indices = np.indices((height, width))
        checkerboard = (row_indices + col_indices) % 2
        update_mask = (checkerboard == (1 if isOdd else 0))
        
        # Add a small visualization of the current checkerboard pattern
        pattern_size = 100
        pattern_img = np.zeros((pattern_size, pattern_size, 3), dtype=np.uint8)
        for i in range(pattern_size):
            for j in range(pattern_size):
                if (i + j) % 2 == (1 if isOdd else 0):
                    pattern_img[i, j] = [0, 0, 255] if channel_idx == 0 else [0, 255, 0] if channel_idx == 1 else [255, 0, 0]
        
        # Add the pattern visualization to the bottom-right corner
        output_frame[height-pattern_size:height, width-pattern_size:width] = pattern_img
        
        return output_frame
    
    def reset(self):
        """
        Reset the effect to its initial state.
        """
        self.persistent_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cycle = 0
