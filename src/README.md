# PF-Compression Library

A library for image transformation, encoding, decoding, and filtering.

## Overview

This library provides a collection of image processing components organized into four main categories:

1. **Encoders**: Convert frames into different representations
2. **Decoders**: Convert encoded data back into frames
3. **Transformers**: Apply combined encoding/decoding operations to frames
4. **Filters**: Apply filtering operations to frames

## Directory Structure

```
src/
└── lib/
    ├── encoder/   # Encoder implementations
    ├── decoder/   # Decoder implementations
    ├── transformer/ # Transformer implementations
    └── filter/    # Filter implementations
```

Each implementation file contains both the abstract base class and concrete implementations.

## Usage

Here's a simple example of using the library:

```python
import cv2
from src.lib.transformer.delta_rgb_pix import DeltaRGBPixTransformer
from src.lib.filter.middle_four_bits import MiddleFourBitsFilter
from src.lib.filter.color_negative import ColorNegativeFilter

# Create instances
delta_rgb_transformer = DeltaRGBPixTransformer(frame_shape=(480, 640, 3))
middle_four_bits_filter = MiddleFourBitsFilter()
color_negative_filter = ColorNegativeFilter()

# Read a frame
frame = cv2.imread('image.jpg')

# Apply transformations
delta_rgb_frame = delta_rgb_transformer.transform(frame)
middle_four_bits_frame = middle_four_bits_filter.filter(frame)
color_negative_frame = color_negative_filter.filter(frame)

# Display results
cv2.imshow('Delta RGB', delta_rgb_frame)
cv2.imshow('Middle Four Bits', middle_four_bits_frame)
cv2.imshow('Color Negative', color_negative_frame)
cv2.waitKey(0)
```

## Extending the Library

To add a new component:

1. Create a new class that inherits from the appropriate base class
2. Implement the required methods
3. Place the implementation in the appropriate directory

For example, to add a new filter:

```python
from src.filter.base import Filter
import numpy as np

class MyNewFilter(Filter):
    def filter(self, frame):
        frame = self.validate_frame(frame)
        # Apply your filtering logic here
        return filtered_frame
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
