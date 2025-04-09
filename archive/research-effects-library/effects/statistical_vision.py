"""
Statistical Vision Effect implementation.
"""
import numpy as np
import cv2
import time
from research_effects_library.core.effect import Effect

class StatisticalVisionEffect(Effect):
    """
    Effect that transforms a frame using statistical color analysis and reconstruction.
    
    This effect:
    1. Analyzes the color distribution of the input frame
    2. Creates histograms of 24-bit colors and individual RGB channels
    3. Reconstructs the frame by sampling from these histograms
    4. Displays analytics about the color distribution
    
    This creates a statistically accurate representation of the original frame
    while significantly reducing the data needed to transmit it.
    """
    
    def __init__(self):
        """
        Initialize the Statistical Vision Effect.
        """
        super().__init__(name="Statistical Vision")
        self.frame_count = 0
        self.last_frame = None
        self.stats = {
            'image_colors': None,
            'image_r': None,
            'image_g': None,
            'image_b': None,
            'image_rgb': None,
            'row_colors': None,
            'row_r': None,
            'row_g': None,
            'row_b': None,
            'col_colors': None,
            'col_r': None,
            'col_g': None,
            'col_b': None,
            'total_pixels': 0,
            'height': 0,
            'width': 0
        }
        self.description = (
            "Analyzes the color distribution of the input frame and reconstructs it "
            "by sampling from 24-bit color and individual RGB channel histograms. "
            "This creates a statistically accurate representation while significantly "
            "reducing the data needed to transmit the frame."
        )
    
    def encode(self, frame):
        """
        Encode the frame by computing the 24-bit color histogram and RGB channel histograms.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            dict: Statistics dictionary with color histograms.
        """
        frame = self.validate_frame(frame)
        height, width = frame.shape[:2]
        
        # Convert to RGB for consistent processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Compute 24-bit color histogram
        pixels = frame_rgb.reshape(-1, 3)
        color_counts = {}
        for pixel in pixels:
            color = tuple(pixel)
            color_counts[color] = color_counts.get(color, 0) + 1
        
        # Compute 8-bit channel histograms
        r_counts = np.bincount(frame_rgb[:, :, 0].flatten(), minlength=256)
        g_counts = np.bincount(frame_rgb[:, :, 1].flatten(), minlength=256)
        b_counts = np.bincount(frame_rgb[:, :, 2].flatten(), minlength=256)
        
        # Update stats dictionary
        self.stats.update({
            'image_colors': color_counts,
            'image_r': r_counts,
            'image_g': g_counts,
            'image_b': b_counts,
            'image_rgb': None,
            'row_colors': None,
            'row_r': None,
            'row_g': None,
            'row_b': None,
            'col_colors': None,
            'col_r': None,
            'col_g': None,
            'col_b': None,
            'total_pixels': height * width,
            'height': height,
            'width': width
        })
        
        return self.stats
    
    def decode(self, stats):
        """
        Decode the frame by sampling colors from the histograms.
        
        Args:
            stats: Dictionary containing color histograms.
            
        Returns:
            np.ndarray: Reconstructed frame.
        """
        height, width = stats['height'], stats['width']
        total_pixels = stats['total_pixels']
        
        # Create working copies of histograms
        color_counts = dict(stats['image_colors'])  # {color: count}
        r_counts = stats['image_r'].copy()  # [0-255] -> count
        g_counts = stats['image_g'].copy()
        b_counts = stats['image_b'].copy()
        
        # Initialize output frame
        output_frame = np.zeros((height, width, 3), dtype=np.uint8)
        pixels_flat = output_frame.reshape(-1, 3)
        
        # Determine sampling strategy (50% 24-bit colors, 50% 8-bit channels)
        total_24bit = total_8bit = total_pixels // 2
        remaining_24bit = total_24bit
        remaining_8bit = total_8bit
        
        # Create a mask for sampling method (0 for 24-bit, 1 for 8-bit)
        method_choices = np.zeros(total_pixels, dtype=np.uint8)
        method_choices[:total_8bit] = 1  # First half for 8-bit
        np.random.shuffle(method_choices)  # Randomize which pixels use which method
        
        # Sample colors
        for i in range(total_pixels):
            if method_choices[i] == 0 and remaining_24bit > 0:
                # Sample from 24-bit color histogram
                if color_counts:  # Check if there are any colors left
                    # Get a random color weighted by frequency
                    colors, counts = zip(*color_counts.items())
                    total_count = sum(counts)
                    if total_count > 0:
                        probs = [count / total_count for count in counts]
                        color_idx = np.random.choice(len(colors), p=probs)
                        color = colors[color_idx]
                        pixels_flat[i] = color
                        
                        # Decrement the count for this color
                        color_counts[color] -= 1
                        if color_counts[color] <= 0:
                            del color_counts[color]
                        
                        remaining_24bit -= 1
                    else:
                        # Switch to 8-bit sampling if no more 24-bit colors
                        method_choices[i] = 1
                else:
                    # Switch to 8-bit sampling if no more 24-bit colors
                    method_choices[i] = 1
            
            if method_choices[i] == 1 and remaining_8bit > 0:
                # Sample from 8-bit channel histograms
                r_total = np.sum(r_counts)
                g_total = np.sum(g_counts)
                b_total = np.sum(b_counts)
                
                if r_total > 0 and g_total > 0 and b_total > 0:
                    r_probs = r_counts / r_total
                    g_probs = g_counts / g_total
                    b_probs = b_counts / b_total
                    
                    r_val = np.random.choice(256, p=r_probs)
                    g_val = np.random.choice(256, p=g_probs)
                    b_val = np.random.choice(256, p=b_probs)
                    
                    pixels_flat[i] = [r_val, g_val, b_val]
                    
                    # Decrement the counts
                    r_counts[r_val] -= 1
                    g_counts[g_val] -= 1
                    b_counts[b_val] -= 1
                    
                    remaining_8bit -= 1
        
        # Convert back to BGR for OpenCV
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        
        return output_frame
    
    def transform(self, frame):
        """
        Transform a frame using the Statistical Vision effect.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        
        # Increment frame counter
        self.frame_count += 1
        
        # Encode the frame
        stats = self.encode(frame)
        
        # Decode the frame
        output_frame = self.decode(stats)
        
        # Store the last frame
        self.last_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        
        # Calculate analytics
        unique_24bit = len(stats['image_colors'])
        unique_r = np.sum(stats['image_r'] > 0)
        unique_g = np.sum(stats['image_g'] > 0)
        unique_b = np.sum(stats['image_b'] > 0)
        
        # Calculate data size
        image_24bit_size = unique_24bit * (3 * 8 + 4 * 8)  # 3 bytes for RGB + 4 bytes for count
        total_bits = image_24bit_size
        
        # Add analytics overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White text
        thickness = 1
        line_spacing = 20
        
        # Add analytics text
        cv2.putText(output_frame, f"Dim: {stats['height']}x{stats['width']}", (10, 20), 
                   font, font_scale, font_color, thickness)
        cv2.putText(output_frame, f"Unique 24-bit: {unique_24bit}", (10, 40), 
                   font, font_scale, font_color, thickness)
        cv2.putText(output_frame, f"Unique R: {unique_r}", (10, 60), 
                   font, font_scale, font_color, thickness)
        cv2.putText(output_frame, f"Unique G: {unique_g}", (10, 80), 
                   font, font_scale, font_color, thickness)
        cv2.putText(output_frame, f"Unique B: {unique_b}", (10, 100), 
                   font, font_scale, font_color, thickness)
        cv2.putText(output_frame, f"Data Size: {total_bits} bits", (10, 120), 
                   font, font_scale, font_color, thickness)
        
        return output_frame
    
    def reset(self):
        """
        Reset the effect to its initial state.
        """
        self.frame_count = 0
        self.last_frame = None
        self.stats = {
            'image_colors': None,
            'image_r': None,
            'image_g': None,
            'image_b': None,
            'image_rgb': None,
            'row_colors': None,
            'row_r': None,
            'row_g': None,
            'row_b': None,
            'col_colors': None,
            'col_r': None,
            'col_g': None,
            'col_b': None,
            'total_pixels': 0,
            'height': 0,
            'width': 0
        }
