"""
RGB Strobe Effect implementation.
"""
import numpy as np
import cv2
from research_effects_library.core.effect import Effect

class RGBStrobeEffect(Effect):
    """
    Effect that transmits one RGB channel per frame in a rotating sequence.
    
    This effect:
    1. Cycles through the RGB channels (R, G, B)
    2. Extracts the current channel from the input frame
    3. Updates the persistent frame with the new channel data
    4. Progressively reconstructs the image over 3 frames
    
    This creates a progressive channel-based transmission effect where the image
    quality improves over time as more channels are transmitted.
    """
    
    def __init__(self):
        """
        Initialize the RGB Strobe Effect.
        """
        super().__init__(name="RGB Strobe")
        self.persistent_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cycle = 0
        self.cycle_order = [0, 1, 2]  # R, G, B
        self.description = (
            "Transmits one RGB channel per frame in a rotating sequence (R, G, B). "
            "This creates a progressive channel-based transmission effect where the image "
            "quality improves over time as more channels are transmitted."
        )
    
    def encode(self, frame):
        """
        Encode the frame by extracting the specified RGB channel.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            tuple: (output_frame, channel_idx)
                - output_frame: NumPy array with only the selected channel.
                - channel_idx: Integer (0=R, 1=G, 2=B) indicating the channel.
        """
        frame = self.validate_frame(frame)
        
        # Get the current channel index
        channel_idx = self.cycle_order[self.cycle]
        
        # Create an output frame with all channels set to 0
        output_frame = np.zeros_like(frame, dtype=np.uint8)
        
        # Copy the selected channel to the output frame
        output_frame[:, :, channel_idx] = frame[:, :, channel_idx]
        
        return output_frame, channel_idx
    
    def decode(self, encoded_data):
        """
        Decode the encoded data by updating the persistent frame with the transmitted channel.
        
        Args:
            encoded_data: tuple (output_frame, channel_idx)
                - output_frame: NumPy array with only the selected channel.
                - channel_idx: Integer (0=R, 1=G, 2=B) indicating the channel.
            
        Returns:
            np.ndarray: Updated persistent frame.
        """
        encoded_frame, channel_idx = encoded_data
        
        # Extract the transmitted channel data
        new_data = encoded_frame[:, :, channel_idx]
        
        # Update the persistent buffer for the selected channel
        self.persistent_frame[:, :, channel_idx] = new_data
        
        return self.persistent_frame.copy()
    
    def transform(self, frame):
        """
        Transform a frame using the RGB Strobe effect.
        
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
        
        # Increment the cycle, wrapping back to 0 after 2
        self.cycle = (self.cycle + 1) % 3
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White text
        thickness = 1
        
        # Add channel information
        channel_names = ["Red", "Green", "Blue"]
        channel_idx = encoded_data[1]
        cv2.putText(output_frame, f"Channel: {channel_names[channel_idx]}", (10, 20), 
                   font, font_scale, font_color, thickness)
        
        # Add data rate information
        frame_height, frame_width = frame.shape[:2]
        bits_per_frame = frame_height * frame_width * 8  # 8 bits per pixel for one channel
        cv2.putText(output_frame, f"Data Rate: {bits_per_frame} bits/frame", (10, 40), 
                   font, font_scale, font_color, thickness)
        
        # Add progress information
        progress = ((self.cycle) % 3) * 33.3
        cv2.putText(output_frame, f"Progress: {progress:.1f}%", (10, 60), 
                   font, font_scale, font_color, thickness)
        
        return output_frame
    
    def reset(self):
        """
        Reset the effect to its initial state.
        """
        self.persistent_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cycle = 0
