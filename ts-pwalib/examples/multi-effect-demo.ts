/**
 * Multi-effect demo for the PF-Compression PWA Library.
 */
import { Effect } from '../src/core/effect';
import {
  ColorNegativeEffect,
  PrimeRGBEffect,
  FibonacciRGBEffect,
  MiddleFourBitsEffect
} from '../src/effects';
import {
  requestMediaAccess,
  createVideoElement
} from '../src/utils/media';
import {
  createCanvas,
  drawVideoToCanvas,
  getImageDataFromCanvas,
  putImageDataOnCanvas
} from '../src/utils/canvas';

// Create effects
const effects: Effect[] = [
  new ColorNegativeEffect(),
  new PrimeRGBEffect(),
  new FibonacciRGBEffect(),
  new MiddleFourBitsEffect(),
  // Add more effects as they are implemented
];

// Set up async function to initialize the app
async function init() {
  try {
    // Request access to the user's camera
    const stream = await requestMediaAccess({ video: true });

    // Create a video element with the stream
    const video = createVideoElement(stream);

    // Wait for the video to be ready
    await new Promise<void>((resolve) => {
      video.onloadedmetadata = () => resolve();
    });

    // Create a canvas with the same dimensions as the video
    const canvas = createCanvas(video.videoWidth, video.videoHeight);

    // Add the canvas to the document
    const container = document.getElementById('video-container');
    if (container) {
      container.appendChild(canvas);
    } else {
      document.body.appendChild(canvas);
    }

    // Set up effect selection
    let currentEffectIndex = -1; // -1 means no effect

    // Create effect selection dropdown
    const effectSelect = document.createElement('select');
    effectSelect.id = 'effect-select';

    // Add "None" option
    const noneOption = document.createElement('option');
    noneOption.value = '-1';
    noneOption.textContent = 'None';
    effectSelect.appendChild(noneOption);

    // Add effect options
    effects.forEach((effect, index) => {
      const option = document.createElement('option');
      option.value = index.toString();
      option.textContent = effect.name;
      effectSelect.appendChild(option);
    });

    // Add the dropdown to the controls
    const controls = document.querySelector('.controls');
    if (controls) {
      // Create a label for the dropdown
      const label = document.createElement('label');
      label.htmlFor = 'effect-select';
      label.textContent = 'Effect: ';

      // Add the label and dropdown to the controls
      controls.prepend(effectSelect);
      controls.prepend(label);
    }

    // Set up event listeners
    effectSelect.addEventListener('change', () => {
      currentEffectIndex = parseInt(effectSelect.value, 10);

      // Reset the effect if one is selected
      if (currentEffectIndex >= 0) {
        effects[currentEffectIndex].reset();
      }
    });

    const resetButton = document.getElementById('reset-effect');
    if (resetButton) {
      resetButton.addEventListener('click', () => {
        if (currentEffectIndex >= 0) {
          effects[currentEffectIndex].reset();
        }
      });
    }

    // Start processing frames
    processFrame();

    function processFrame() {
      // Draw the video frame onto the canvas
      drawVideoToCanvas(video, canvas);

      // Get the ImageData from the canvas
      const imageData = getImageDataFromCanvas(canvas);

      // Apply the effect if one is selected
      let transformedImageData = imageData;
      if (currentEffectIndex >= 0) {
        transformedImageData = effects[currentEffectIndex].transform(imageData);
      }

      // Put the transformed ImageData back onto the canvas
      putImageDataOnCanvas(canvas, transformedImageData);

      // Request the next frame
      requestAnimationFrame(processFrame);
    }
  } catch (error) {
    console.error('Error initializing app:', error);
  }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);
