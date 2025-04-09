"""
Prime RGB Effect implementation.
"""
import numpy as np
from research_effects_library.core.effect import Effect
from research_effects_library.core.utils import precompute_nearest_primes

class PrimeRGBEffect(Effect):
    """
    Effect that maps each RGB value to the nearest prime number.
    """

    def __init__(self, max_value=255):
        """
        Initialize the Prime RGB Effect.

        Args:
            max_value: Maximum value for the lookup table.
        """
        super().__init__(name="Prime RGB")
        self.max_value = max_value
        self.prime_lookup = precompute_nearest_primes(max_value)

    def transform(self, frame):
        """
        Transform a frame by mapping each RGB value to the nearest prime number.

        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).

        Returns:
            transformed_frame: NumPy array of shape (height, width, 3) with uint8 values (BGR or RGB).
        """
        frame = self.validate_frame(frame)

        # Apply the prime lookup table to the frame
        output = self.prime_lookup[frame]

        return output
