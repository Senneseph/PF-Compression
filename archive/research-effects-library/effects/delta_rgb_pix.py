"""
Delta RGB Pixel Effect implementation.
"""
import numpy as np
import cv2
from lib.effect.base import Effect

class DeltaRGBPixEffect(Effect):
    """
    Effect that identifies changed RGB values compared to a persistent frame.
    
    This effect:
    1. Encodes the frame by identifying changed RGB values compared to a persistent frame.
    2. Optionally denoises the encoded data to remove small changes.
    3. Decodes the encoded data back into a frame.
    """
    
    def __init__(self, frame_shape=(480, 640, 3), denoise_threshold=15):
        """
        Initialize the Delta RGB Pixel Effect.
        
        Args:
            frame_shape: Shape of the frame (height, width, channels).
            denoise_threshold: Threshold for denoising (default: 15).
        """
        super().__init__(name="Delta RGB Pixel")
        self.frame_shape = frame_shape
        self.denoise_threshold = denoise_threshold
        self.persistent_frame = np.zeros(frame_shape, dtype=np.uint8)
    
    def encode(self, frame):
        """
        Encode the frame by identifying changed RGB values compared to the persistent frame.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            tuple: (mask, new_values)
                - mask: 2D array of 3-bit values (0-7) indicating which channels changed.
                - new_values: 3D array of new RGB values for the changed channels.
        """
        frame = self.validate_frame(frame, expected_shape=self.frame_shape)
        
        # Compare the current frame with the persistent frame
        changed = frame != self.persistent_frame  # Shape: (height, width, 3), dtype: bool
        
        # Create a 3-bit mask indicating which channels changed
        # Bit 2 (4): R channel changed
        # Bit 1 (2): G channel changed
        # Bit 0 (1): B channel changed
        mask = (changed[:, :, 0].astype(np.uint8) * 4 +
                changed[:, :, 1].astype(np.uint8) * 2 +
                changed[:, :, 2].astype(np.uint8) * 1)
        
        # Create a 3D array of new RGB values for the changed channels
        new_values = np.zeros_like(frame)
        new_values[changed] = frame[changed]
        
        # Update the persistent frame
        self.persistent_frame = frame.copy()
        
        # Denoise the encoded data
        return self._denoise(mask, new_values, frame, self.denoise_threshold)
    
    def _denoise(self, mask, new_values, frame, threshold=1):
        """
        Denoise the output of encode by excluding RGB changes below a threshold.
        
        Args:
            mask: 2D array of 3-bit values (0-7) indicating which channels changed.
            new_values: 3D array of new RGB values for the changed channels.
            frame: The original input frame.
            threshold: The maximum allowed difference to consider a change as noise (default: 1).
            
        Returns:
            tuple: (updated_mask, new_values)
                - updated_mask: 2D array with small changes excluded.
                - new_values: Original new_values (unchanged).
        """
        # Compute the absolute difference between the frame and persistent frame
        diff = np.abs(frame.astype(np.int16) - self.persistent_frame.astype(np.int16))
        # Shape: (height, width, 3), dtype: int16
        
        # Identify channels where the difference is <= threshold (noise)
        is_noise = diff <= threshold  # Shape: (height, width, 3), dtype: bool
        
        # Extract the current channel-specific changes from the mask
        r_changed = (mask & 4) > 0  # Bit 2: R channel
        g_changed = (mask & 2) > 0  # Bit 1: G channel
        b_changed = (mask & 1) > 0  # Bit 0: B channel
        
        # Clear the bits for channels where the change is considered noise
        r_changed = r_changed & ~is_noise[:, :, 0]
        g_changed = g_changed & ~is_noise[:, :, 1]
        b_changed = b_changed & ~is_noise[:, :, 2]
        
        # Recompute the mask with the filtered changes
        updated_mask = (r_changed.astype(np.uint8) * 4 +
                        g_changed.astype(np.uint8) * 2 +
                        b_changed.astype(np.uint8) * 1)
        
        return updated_mask, new_values
    
    def decode(self, encoded_data):
        """
        Decode the frame by applying the changed RGB values to the persistent frame.
        
        Args:
            encoded_data: tuple (mask, new_values)
                - mask: 2D array of 3-bit values (0-7) indicating which channels changed.
                - new_values: 3D array of new RGB values for the changed channels.
            
        Returns:
            np.ndarray: Updated persistent frame.
        """
        mask, new_values = encoded_data
        
        # Extract channel-specific masks
        r_changed = (mask & 4) > 0  # Bit 2: R channel changed
        g_changed = (mask & 2) > 0  # Bit 1: G channel changed
        b_changed = (mask & 1) > 0  # Bit 0: B channel changed
        # Each is shape: (height, width), dtype: bool
        
        # Update the persistent frame with the new values
        if np.any(r_changed):
            self.persistent_frame[r_changed, 0] = new_values[r_changed, 0]
        if np.any(g_changed):
            self.persistent_frame[g_changed, 1] = new_values[g_changed, 1]
        if np.any(b_changed):
            self.persistent_frame[b_changed, 2] = new_values[b_changed, 2]
        
        # Calculate statistics
        total_pixels = self.frame_shape[0] * self.frame_shape[1]
        changed_pixels = np.count_nonzero(mask)
        changed_channels = np.count_nonzero(r_changed) + np.count_nonzero(g_changed) + np.count_nonzero(b_changed)
        
        # Create a copy of the persistent frame for annotation
        output_frame = self.persistent_frame.copy()
        
        # Overlay statistics on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White text
        thickness = 1
        
        # Add text with statistics
        cv2.putText(output_frame, f"Changed Pixels: {changed_pixels}/{total_pixels} ({changed_pixels/total_pixels:.1%})",
                   (10, 20), font, font_scale, color, thickness)
        cv2.putText(output_frame, f"Changed Channels: {changed_channels}/{total_pixels*3} ({changed_channels/(total_pixels*3):.1%})",
                   (10, 40), font, font_scale, color, thickness)
        
        return output_frame
    
    def transform(self, frame):
        """
        Transform a frame using the Delta RGB Pixel transformation.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame, expected_shape=self.frame_shape)
        
        # Encode the frame
        encoded_data = self.encode(frame)
        
        # Decode the encoded data
        output_frame = self.decode(encoded_data)
        
        return output_frame
    
    def reset(self):
        """
        Reset the persistent frame to all zeros.
        """
        self.persistent_frame = np.zeros(self.frame_shape, dtype=np.uint8)
