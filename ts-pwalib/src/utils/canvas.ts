/**
 * Utility functions for working with canvas elements.
 */

/**
 * Create a canvas element with the specified dimensions.
 * 
 * @param width - The width of the canvas.
 * @param height - The height of the canvas.
 * @returns A canvas element with the specified dimensions.
 */
export function createCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/**
 * Draw a video frame onto a canvas.
 * 
 * @param video - The video element to draw.
 * @param canvas - The canvas element to draw onto.
 */
export function drawVideoToCanvas(video: HTMLVideoElement, canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Could not get canvas context');
  }
  
  // Ensure the canvas has the same dimensions as the video
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  
  // Draw the video frame onto the canvas
  ctx.drawImage(video, 0, 0);
}

/**
 * Get the ImageData from a canvas.
 * 
 * @param canvas - The canvas element to get the ImageData from.
 * @returns The ImageData from the canvas.
 */
export function getImageDataFromCanvas(canvas: HTMLCanvasElement): ImageData {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Could not get canvas context');
  }
  
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/**
 * Put ImageData onto a canvas.
 * 
 * @param canvas - The canvas element to put the ImageData onto.
 * @param imageData - The ImageData to put onto the canvas.
 */
export function putImageDataOnCanvas(canvas: HTMLCanvasElement, imageData: ImageData): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Could not get canvas context');
  }
  
  ctx.putImageData(imageData, 0, 0);
}

/**
 * Create an ImageData object with the specified dimensions.
 * 
 * @param width - The width of the ImageData.
 * @param height - The height of the ImageData.
 * @returns A new ImageData object with the specified dimensions.
 */
export function createImageData(width: number, height: number): ImageData {
  // Create a temporary canvas to get a context
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Could not get canvas context');
  }
  
  return ctx.createImageData(width, height);
}
