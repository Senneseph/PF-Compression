import { Effect } from '../core/effect';
/**
 * Effect that extracts the middle 4 bits from each RGB component.
 *
 * This effect:
 * 1. Extracts bits 2-5 (0-indexed) from each RGB component
 * 2. Shifts them to the right position (bits 0-3)
 * 3. Preserves the alpha channel
 */
export class MiddleFourBitsEffect extends Effect {
    /**
     * Initialize the Middle Four Bits Effect.
     */
    constructor() {
        super('Middle Four Bits', 'Extracts the middle 4 bits from each RGB component, creating a posterized effect.');
    }
    /**
     * Transform an ImageData object by extracting the middle 4 bits from each RGB component.
     *
     * @param imageData - The ImageData object to transform.
     * @returns The transformed ImageData object.
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
            // Extract the middle 4 bits (bits 2-5) from each RGB component
            // Shift them to the right position (bits 0-3)
            // Multiply by 16 to scale to 0-255 range
            outputData[i] = ((inputData[i] >> 2) & 0x0F) * 16; // R
            outputData[i + 1] = ((inputData[i + 1] >> 2) & 0x0F) * 16; // G
            outputData[i + 2] = ((inputData[i + 2] >> 2) & 0x0F) * 16; // B
            // Preserve the alpha channel
            outputData[i + 3] = inputData[i + 3]; // A
        }
        return output;
    }
}
