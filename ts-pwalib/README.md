# PF-Compression PWA Library

This library provides a TypeScript implementation of the PF-Compression effects for use in Progressive Web Applications.

## Installation

```bash
npm install pf-compression-pwa
```

## Usage

```typescript
import { ColorNegativeEffect } from 'pf-compression-pwa';
import { drawVideoToCanvas, getImageDataFromCanvas, putImageDataOnCanvas } from 'pf-compression-pwa';

// Create an effect
const effect = new ColorNegativeEffect();

// Set up video and canvas elements
const video = document.getElementById('video') as HTMLVideoElement;
const canvas = document.getElementById('canvas') as HTMLCanvasElement;

// Process a frame
function processFrame() {
  // Draw the video frame onto the canvas
  drawVideoToCanvas(video, canvas);
  
  // Get the ImageData from the canvas
  const imageData = getImageDataFromCanvas(canvas);
  
  // Apply the effect
  const transformedImageData = effect.transform(imageData);
  
  // Put the transformed ImageData back onto the canvas
  putImageDataOnCanvas(canvas, transformedImageData);
  
  // Request the next frame
  requestAnimationFrame(processFrame);
}

// Start processing frames
processFrame();
```

## Available Effects

- `ColorNegativeEffect`: Inverts the colors of an image.
- More effects coming soon!

## Development

### Prerequisites

- Node.js 14+
- npm 7+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pf-compression-pwa.git

# Install dependencies
cd pf-compression-pwa
npm install

# Build the library
npm run build
```

### Testing

```bash
npm test
```

## License

MIT
