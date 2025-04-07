"""
Even/Odd Color Transformer implementation.
"""
import numpy as np
from abc import ABC, abstractmethod
from src.lib.encoder.even_odd_color import EvenOddColorEncoder
from src.lib.decoder.even_odd_color import EvenOddColorDecoder

class Transformer(ABC):
    """
    Abstract base class for all transformers.
    
    A transformer takes an input frame, applies a transformation, and returns a transformed frame.
    It may use an encoder and decoder internally, or apply the transformation directly.
    """
    
    @abstractmethod
    def transform(self, frame):
        """
        Transform a frame.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        pass
    
    def validate_frame(self, frame, expected_shape=None, expected_dtype=np.uint8):
        """
        Validate that the frame has the expected shape and data type.
        
        Args:
            frame: NumPy array to validate.
            expected_shape: Tuple of expected shape (height, width, channels) or None.
            expected_dtype: Expected data type of the frame.
            
        Returns:
            frame: The validated frame, converted to the expected data type if necessary.
            
        Raises:
            ValueError: If the frame does not have the expected shape.
        """
        # Ensure the frame is a NumPy array
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a NumPy array")
        
        # Ensure the frame has the expected shape
        if expected_shape is not None:
            if frame.shape != expected_shape:
                raise ValueError(f"Frame must have shape {expected_shape}, got {frame.shape}")
        
        # Ensure the frame has the expected data type
        if frame.dtype != expected_dtype:
            frame = frame.astype(expected_dtype)
        
        return frame

class EvenOddColorTransformer(Transformer):
    """
    Transformer that applies the Even/Odd Color transformation to a frame.
    
    This transformer:
    1. Alternates between forcing odd and even values each frame.
    2. Encodes the frame with the current parity mode.
    3. Decodes the frame into a persistent buffer for progressive reconstruction.
    """
    
    def __init__(self, frame_shape=(480, 640, 3), initial_odd=True):
        """
        Initialize the Even/Odd Color Transformer.
        
        Args:
            frame_shape: Shape of the frame (height, width, channels).
            initial_odd: Whether to start with odd values (True) or even values (False).
        """
        self.frame_shape = frame_shape
        
        # Initialize the encoder and decoder
        self.encoder = EvenOddColorEncoder(initial_odd)
        self.decoder = EvenOddColorDecoder(frame_shape)
    
    def transform(self, frame):
        """
        Transform a frame using the Even/Odd Color transformation.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
            
        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame, expected_shape=self.frame_shape)
        
        # Encode the frame
        encoded_data = self.encoder.encode(frame)
        
        # Decode the encoded data
        output_frame = self.decoder.decode(encoded_data)
        
        return output_frame
    
    def reset(self):
        """
        Reset the transformer to its initial state.
        """
        self.encoder.odd_cycle = True
        self.decoder.reset()
