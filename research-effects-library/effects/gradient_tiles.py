"""
Gradient Tiles Effect implementation.
"""
import numpy as np
import cv2
from research_effects_library.core.effect import Effect

class GradientTilesEffect(Effect):
    """
    Effect that divides the frame into tiles and creates smooth gradients between them.
    
    This effect:
    1. Divides the frame into a grid of tiles (e.g., 20x20)
    2. Extracts a reference color for each tile
    3. Computes adjustments based on neighboring tiles
    4. Reconstructs the frame with smooth gradients between tiles
    
    This creates a stylized, vectorized look with smooth color transitions.
    """
    
    def __init__(self, tile_width=32, tile_height=24):
        """
        Initialize the Gradient Tiles Effect.
        
        Args:
            tile_width: Width of each tile in pixels (default: 32).
            tile_height: Height of each tile in pixels (default: 24).
        """
        super().__init__(name="Gradient Tiles")
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.description = (
            "Divides the frame into tiles and creates smooth gradients between them. "
            "This creates a stylized, vectorized look with smooth color transitions, "
            "similar to vector-based compression techniques."
        )
    
    def encode(self, frame):
        """
        Encode a frame into reference colors and neighbor adjustments for tiles.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            tuple: (ref_colors, adjustments, metadata)
                - ref_colors: NumPy array of reference colors for each tile.
                - adjustments: NumPy array of adjustments for neighboring tiles.
                - metadata: Dictionary with tile information.
        """
        frame = self.validate_frame(frame)
        height, width = frame.shape[:2]
        
        # Calculate number of tiles
        num_tiles_y = height // self.tile_height
        num_tiles_x = width // self.tile_width
        
        # Extract reference color for each tile (mean color)
        ref_colors = np.zeros((num_tiles_y, num_tiles_x, 3), dtype=np.uint8)
        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                y_start = ty * self.tile_height
                x_start = tx * self.tile_width
                tile = frame[y_start:y_start+self.tile_height, x_start:x_start+self.tile_width]
                ref_colors[ty, tx] = np.mean(tile, axis=(0, 1)).astype(np.uint8)
        
        # Compute adjustments for neighboring tiles
        # 4 neighbors: top, right, bottom, left
        neighbor_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        adjustments = np.zeros((num_tiles_y, num_tiles_x, 4), dtype=np.uint8)
        
        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                ref_color = ref_colors[ty, tx]
                
                for n_idx, (dy, dx) in enumerate(neighbor_offsets):
                    ny, nx = ty + dy, tx + dx
                    if 0 <= ny < num_tiles_y and 0 <= nx < num_tiles_x:
                        neighbor_ref = ref_colors[ny, nx]
                        diff = (neighbor_ref.astype(np.int16) - ref_color.astype(np.int16))
                        adjustment = np.clip(np.mean(diff) // 16, -8, 7) + 8
                        adjustments[ty, tx, n_idx] = adjustment.astype(np.uint8)
                    else:
                        adjustments[ty, tx, n_idx] = 8  # Neutral adjustment for edges
        
        metadata = {
            'tile_width': self.tile_width,
            'tile_height': self.tile_height,
            'num_tiles_y': num_tiles_y,
            'num_tiles_x': num_tiles_x
        }
        
        return ref_colors, adjustments, metadata
    
    def decode(self, encoded_data):
        """
        Decode reference colors and adjustments into a full frame with vectorized gradients.
        
        Args:
            encoded_data: tuple (ref_colors, adjustments, metadata)
                - ref_colors: NumPy array of reference colors for each tile.
                - adjustments: NumPy array of adjustments for neighboring tiles.
                - metadata: Dictionary with tile information.
            
        Returns:
            np.ndarray: Reconstructed frame with gradients.
        """
        ref_colors, adjustments, metadata = encoded_data
        
        # Extract metadata
        tile_width = metadata['tile_width']
        tile_height = metadata['tile_height']
        num_tiles_y = metadata['num_tiles_y']
        num_tiles_x = metadata['num_tiles_x']
        
        # Calculate output dimensions
        height = num_tiles_y * tile_height
        width = num_tiles_x * tile_width
        
        # Create output frame
        output = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Neighbor offsets (top, right, bottom, left)
        neighbor_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # For each tile, create a gradient based on reference color and adjustments
        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                y_start = ty * tile_height
                x_start = tx * tile_width
                
                # Get reference color for this tile
                ref_color = ref_colors[ty, tx].astype(np.float32)
                
                # Create gradient weights for each pixel in the tile
                y_coords, x_coords = np.mgrid[0:tile_height, 0:tile_width]
                
                # Normalize coordinates to [0, 1] range
                y_norm = y_coords / tile_height
                x_norm = x_coords / tile_width
                
                # Calculate weights for each neighbor
                top_weight = 1.0 - y_norm
                right_weight = x_norm
                bottom_weight = y_norm
                left_weight = 1.0 - x_norm
                
                # Combine weights
                weights = np.stack([
                    top_weight, right_weight, bottom_weight, left_weight
                ], axis=-1)
                
                # Normalize weights to sum to 1
                weights = weights / np.sum(weights, axis=-1, keepdims=True)
                
                # Get adjustments for this tile
                tile_adjustments = adjustments[ty, tx].astype(np.float32)
                
                # Convert adjustments from [0, 16] to [-8, 8]
                tile_adjustments = (tile_adjustments - 8) * 16
                
                # Calculate color adjustment for each pixel
                color_adjustment = np.sum(weights * tile_adjustments.reshape(1, 1, 4), axis=-1)
                
                # Apply adjustment to reference color
                for c in range(3):
                    output[y_start:y_start+tile_height, x_start:x_start+tile_width, c] = np.clip(
                        ref_color[c] + color_adjustment, 0, 255
                    ).astype(np.uint8)
        
        return output
    
    def transform(self, frame):
        """
        Transform a frame using the Gradient Tiles effect.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)
        
        # Encode the frame
        encoded_data = self.encode(frame)
        ref_colors, adjustments, metadata = encoded_data
        
        # Compute analytics
        tile_area = f"{metadata['tile_height']}x{metadata['tile_width']}"
        tile_count = metadata['num_tiles_y'] * metadata['num_tiles_x']
        tile_bit_size = 24 + (8 * 4)  # 24 bits for color + 8 bits for each of 4 adjustments
        total_size_bits = tile_count * tile_bit_size
        frame_rate = 30  # Assuming 30 fps
        data_rate_kbps = (total_size_bits * frame_rate) // 1000
        
        # Decode the frame
        output_frame = self.decode(encoded_data)
        
        # Overlay analytics text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White text
        thickness = 1
        line_spacing = 20
        
        # Add analytics text
        cv2.putText(output_frame, f"Tile Area: {tile_area}", (10, 20), 
                   font, font_scale, font_color, thickness)
        cv2.putText(output_frame, f"Tile Count: {tile_count}", (10, 40), 
                   font, font_scale, font_color, thickness)
        cv2.putText(output_frame, f"Tile Size: {tile_bit_size} bits", (10, 60), 
                   font, font_scale, font_color, thickness)
        cv2.putText(output_frame, f"Total Size: {total_size_bits} bits", (10, 80), 
                   font, font_scale, font_color, thickness)
        cv2.putText(output_frame, f"Data Rate: {data_rate_kbps} kbps", (10, 100), 
                   font, font_scale, font_color, thickness)
        
        return output_frame
