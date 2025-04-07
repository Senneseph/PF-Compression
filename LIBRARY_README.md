# PF-Compression Library

A library for image transformation, encoding, decoding, and filtering.

## Project Structure

This project is organized into three main parts:

1. **Abstract Library** (`/lib`): Core abstract classes and interfaces
2. **Effects Library** (`/research-effects-library`): Concrete implementations of effects
3. **Webcam Application**: The original application using the effects

```
PF-Compression/
├── lib/                      # Abstract library
│   ├── effect/               # Abstract effect interfaces
│   │   ├── __init__.py
│   │   └── base.py           # Effect base class
│   ├── __init__.py
│   └── utils.py              # Utility functions
│
├── research-effects-library/  # Effects library
│   ├── effects/              # Concrete effect implementations
│   │   ├── __init__.py
│   │   ├── delta_rgb_pix.py
│   │   ├── prime_rgb.py
│   │   ├── fibonacci_rgb.py
│   │   ├── color_negative.py
│   │   ├── middle_four_bits.py
│   │   └── panavision_1970s.py
│   └── __init__.py
│
├── examples/                 # Example scripts
│   ├── basic_usage.py
│   └── effects_demo.py
│
└── webcam.py                 # Original webcam application
```

## Abstract Library

The abstract library provides the core interfaces and base classes for the project:

- `Effect`: Base class for all effects, combining encoding, decoding, filtering, and transformation operations

## Effects Library

The effects library contains concrete implementations of various effects:

- `DeltaRGBPixEffect`: Identifies changed RGB values compared to a persistent frame
- `PrimeRGBEffect`: Maps each RGB value to the nearest prime number
- `FibonacciRGBEffect`: Maps each RGB value to the nearest Fibonacci number
- `ColorNegativeEffect`: Inverts each channel to create a negative effect
- `MiddleFourBitsEffect`: Preserves only the middle four bits of each pixel
- `Panavision1970sEffect`: Emulates 1970s Panavision color grading

## Usage

Here's a simple example of using the library:

```python
import cv2
from research_effects_library.effects.prime_rgb import PrimeRGBEffect

# Create an effect
effect = PrimeRGBEffect()

# Read a frame
frame = cv2.imread('image.jpg')

# Apply the effect
transformed_frame = effect.transform(frame)

# Display the result
cv2.imshow('Prime RGB', transformed_frame)
cv2.waitKey(0)
```

## Running the Examples

To run the examples:

```
python examples/effects_demo.py
```

## Extending the Library

To add a new effect:

1. Create a new class that inherits from `Effect`
2. Implement the required methods (`transform` at minimum)
3. Place the implementation in the `research-effects-library/effects` directory

For example:

```python
from lib.effect.base import Effect
import numpy as np

class MyNewEffect(Effect):
    def __init__(self):
        super().__init__(name="My New Effect")
    
    def transform(self, frame):
        frame = self.validate_frame(frame)
        # Apply your transformation logic here
        return transformed_frame
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
