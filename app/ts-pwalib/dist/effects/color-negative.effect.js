import { Effect } from '../core/effect';
/**
 * Effect that inverts the colors of an image.
 *
 * This effect:
 * 1. Inverts each RGB component (255 - value)
 * 2. Preserves the alpha channel
 */
export class ColorNegativeEffect extends Effect {
    /**
     * Initialize the Color Negative Effect.
     */
    constructor() {
        super('Color Negative', 'Inverts the colors of an image by subtracting each RGB component from 255.');
    }
    /**
     * Transform an ImageData object by inverting its colors.
     *
     * @param imageData - The ImageData object to transform.
     * @returns The transformed ImageData object with inverted colors.
     */
    transform(imageData) {
        // Validate the input
        this.validateImageData(imageData);
        // Create a new ImageData object for the output
        const output = new ImageData(imageData.width, imageData.height);
        // Get the data arrays
        const inputData = imageData.data;
        const outputData = output.data;
        // Process each pixel
        for (let i = 0; i < inputData.length; i += 4) {
            // Invert the RGB components
            outputData[i] = 255 - inputData[i]; // R
            outputData[i + 1] = 255 - inputData[i + 1]; // G
            outputData[i + 2] = 255 - inputData[i + 2]; // B
            // Preserve the alpha channel
            outputData[i + 3] = inputData[i + 3]; // A
        }
        return output;
    }
}
