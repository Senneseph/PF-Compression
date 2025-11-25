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
export declare function createCanvas(width: number, height: number): HTMLCanvasElement;
/**
 * Draw a video frame onto a canvas.
 *
 * @param video - The video element to draw.
 * @param canvas - The canvas element to draw onto.
 */
export declare function drawVideoToCanvas(video: HTMLVideoElement, canvas: HTMLCanvasElement): void;
/**
 * Get the ImageData from a canvas.
 *
 * @param canvas - The canvas element to get the ImageData from.
 * @returns The ImageData from the canvas.
 */
export declare function getImageDataFromCanvas(canvas: HTMLCanvasElement): ImageData;
/**
 * Put ImageData onto a canvas.
 *
 * @param canvas - The canvas element to put the ImageData onto.
 * @param imageData - The ImageData to put onto the canvas.
 */
export declare function putImageDataOnCanvas(canvas: HTMLCanvasElement, imageData: ImageData): void;
/**
 * Create an ImageData object with the specified dimensions.
 *
 * @param width - The width of the ImageData.
 * @param height - The height of the ImageData.
 * @returns A new ImageData object with the specified dimensions.
 */
export declare function createImageData(width: number, height: number): ImageData;
