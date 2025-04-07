"""
Bit Depth Strobe Effect implementation.
"""
import numpy as np
import cv2
from research_effects_library.core.effect import Effect

class BitDepthStrobeEffect(Effect):
    """
    Effect that transmits one bit layer per frame for all RGB channels.
    
    This effect:
    1. Cycles through the 8 bits (0-7) of each RGB channel
    2. Extracts the current bit layer from all channels
    3. Updates the persistent frame with the new bit layer
    4. Progressively reconstructs the image over 8 frames
    
    This creates a progressive bit-plane transmission effect where the image
    quality improves over time as more significant bits are transmitted.
    """
    
    def __init__(self):
        """
        Initialize the Bit Depth Strobe Effect.
        """
        super().__init__(name="Bit Depth Strobe")
        self.persistent_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.current_bit = 0
        self.description = (
            "Transmits one bit layer per frame for all RGB channels, cycling through "
            "all 8 bits (0-7). This creates a progressive bit-plane transmission effect "
            "where the image quality improves over time as more significant bits are transmitted."
        )
    
    def encode(self, frame):
        """
        Encode the frame by extracting the current bit layer for all RGB channels.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            tuple: (bit_layer, mask)
                - bit_layer: Integer (0-7) indicating the current bit being transmitted.
                - mask: NumPy array of shape (height, width, 3) with 1-bit values (0 or 1) for R, G, B.
        """
        frame = self.validate_frame(frame)
        
        # Extract the current bit layer from all channels
        bit_layer = self.current_bit
        mask = (frame >> bit_layer) & 1  # Shape: (height, width, 3), values 0 or 1
        
        return bit_layer, mask
    
    def decode(self, encoded_data):
        """
        Decode the bit layer into the persistent frame, replacing the specified bit.
        
        Args:
            encoded_data: tuple (bit_layer, mask)
                - bit_layer: Integer (0-7) indicating the bit being updated.
                - mask: NumPy array of shape (height, width, 3) with 1-bit values (0 or 1).
            
        Returns:
            np.ndarray: Updated persistent frame with the new bit layer applied.
        """
        bit_layer, mask = encoded_data
        
        # Compute bit_mask directly in uint8 to avoid deprecation warning
        bit_mask = (~np.uint8(1 << bit_layer)).astype(np.uint8)  # e.g., ~0b100 = 0b11111011 = 251
        
        # Clear the current bit layer in the persistent frame
        self.persistent_frame = (self.persistent_frame & bit_mask).astype(np.uint8)
        
        # Set the new bit layer
        self.persistent_frame = (self.persistent_frame | (mask << bit_layer)).astype(np.uint8)
        
        return self.persistent_frame.copy()
    
    def transform(self, frame):
        """
        Transform a frame using the Bit Depth Strobe effect.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        
        # Encode the current bit layer
        encoded_data = self.encode(frame)
        
        # Decode into the persistent frame
        output_frame = self.decode(encoded_data)
        
        # Increment the bit layer, cycling back to 0 after 7
        self.current_bit = (self.current_bit + 1) % 8
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White text
        thickness = 1
        
        # Add bit layer information
        cv2.putText(output_frame, f"Bit Layer: {encoded_data[0]}", (10, 20), 
                   font, font_scale, font_color, thickness)
        
        # Add data rate information
        frame_height, frame_width = frame.shape[:2]
        bits_per_frame = frame_height * frame_width * 3  # One bit per RGB channel per pixel
        cv2.putText(output_frame, f"Data Rate: {bits_per_frame} bits/frame", (10, 40), 
                   font, font_scale, font_color, thickness)
        
        # Add progress information
        progress = (self.current_bit / 8) * 100
        cv2.putText(output_frame, f"Progress: {progress:.1f}%", (10, 60), 
                   font, font_scale, font_color, thickness)
        
        return output_frame
    
    def reset(self):
        """
        Reset the effect to its initial state.
        """
        self.persistent_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.current_bit = 0
