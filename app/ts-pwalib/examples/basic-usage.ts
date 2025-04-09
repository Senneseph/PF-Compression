/**
 * Basic usage example for the PF-Compression PWA Library.
 */
import { ColorNegativeEffect } from '../src/effects/color-negative.effect';
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

// Create an effect
const effect = new ColorNegativeEffect();

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
    document.body.appendChild(canvas);
    
    // Start processing frames
    processFrame();
    
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
  } catch (error) {
    console.error('Error initializing app:', error);
  }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);
