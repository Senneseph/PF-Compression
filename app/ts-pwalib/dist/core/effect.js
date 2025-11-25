/**
 * Abstract base class for all effects.
 *
 * An effect combines encoding, decoding, filtering, and transformation operations
 * into a single cohesive unit. It provides a standard interface for all effects
 * in the library.
 */
export class Effect {
    /**
     * Initialize the effect.
     *
     * @param name - Optional name for the effect.
     * @param description - Optional description of the effect.
     */
    constructor(name, description) {
        this.name = name || this.constructor.name;
        this.description = description || '';
    }
    /**
     * Encode an ImageData object into a different representation.
     *
     * This method should be overridden by effects that need to encode frames.
     *
     * @param imageData - The ImageData object to encode.
     * @returns The encoded data in a format specific to the effect.
     */
    encode(imageData) {
        return imageData;
    }
    /**
     * Decode encoded data back into an ImageData object.
     *
     * This method should be overridden by effects that need to decode frames.
     *
     * @param encodedData - The encoded data in a format specific to the effect.
     * @returns The decoded ImageData object.
     */
    decode(encodedData) {
        return encodedData;
    }
    /**
     * Apply a filtering operation to an ImageData object.
     *
     * This method should be overridden by effects that need to filter frames.
     *
     * @param imageData - The ImageData object to filter.
     * @returns The filtered ImageData object.
     */
    filter(imageData) {
        return imageData;
    }
    /**
     * Reset the effect to its initial state.
     *
     * This method should be overridden by effects that maintain state.
     */
    reset() {
        // Default implementation does nothing
    }
    /**
     * Validate that the ImageData object has the expected dimensions.
     *
     * @param imageData - The ImageData object to validate.
     * @param expectedWidth - The expected width of the ImageData object.
     * @param expectedHeight - The expected height of the ImageData object.
     * @returns The validated ImageData object.
     * @throws Error if the ImageData object does not have the expected dimensions.
     */
    validateImageData(imageData, expectedWidth, expectedHeight) {
        if (!(imageData instanceof ImageData)) {
            throw new Error('Input must be an ImageData object');
        }
        if (expectedWidth !== undefined && imageData.width !== expectedWidth) {
            throw new Error(`ImageData must have width ${expectedWidth}, got ${imageData.width}`);
        }
        if (expectedHeight !== undefined && imageData.height !== expectedHeight) {
            throw new Error(`ImageData must have height ${expectedHeight}, got ${imageData.height}`);
        }
        return imageData;
    }
}
