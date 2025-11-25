/**
 * Filters for the PF-Compression PWA
 *
 * This module provides filter functions that modify video frames.
 * These are TypeScript equivalents of the filter_* functions in webcam.py.
 */
/**
 * Base class for filters
 */
export declare abstract class Filter {
    /**
     * Apply the filter to an ImageData object
     *
     * @param imageData - ImageData to filter
     * @returns Filtered ImageData
     */
    abstract apply(imageData: ImageData): ImageData;
}
/**
 * Color Negative filter
 *
 * Inverts the colors of the image.
 */
export declare class ColorNegativeFilter extends Filter {
    /**
     * Apply the color negative filter to an ImageData object
     *
     * @param imageData - ImageData to filter
     * @returns Filtered ImageData
     */
    apply(imageData: ImageData): ImageData;
}
/**
 * Middle Four Bits filter
 *
 * Extracts the middle 4 bits from each RGB component.
 */
export declare class MiddleFourBitsFilter extends Filter {
    /**
     * Apply the middle four bits filter to an ImageData object
     *
     * @param imageData - ImageData to filter
     * @returns Filtered ImageData
     */
    apply(imageData: ImageData): ImageData;
}
/**
 * Grayscale filter
 *
 * Converts the image to grayscale.
 */
export declare class GrayscaleFilter extends Filter {
    /**
     * Apply the grayscale filter to an ImageData object
     *
     * @param imageData - ImageData to filter
     * @returns Filtered ImageData
     */
    apply(imageData: ImageData): ImageData;
}
/**
 * Threshold filter
 *
 * Applies a threshold to the image, converting it to black and white.
 */
export declare class ThresholdFilter extends Filter {
    private threshold;
    /**
     * Create a new Threshold filter
     *
     * @param threshold - Threshold value (0-255)
     */
    constructor(threshold?: number);
    /**
     * Set the threshold value
     *
     * @param threshold - Threshold value (0-255)
     */
    setThreshold(threshold: number): void;
    /**
     * Apply the threshold filter to an ImageData object
     *
     * @param imageData - ImageData to filter
     * @returns Filtered ImageData
     */
    apply(imageData: ImageData): ImageData;
}
/**
 * Brightness filter
 *
 * Adjusts the brightness of the image.
 */
export declare class BrightnessFilter extends Filter {
    private brightness;
    /**
     * Create a new Brightness filter
     *
     * @param brightness - Brightness adjustment (-255 to 255)
     */
    constructor(brightness?: number);
    /**
     * Set the brightness value
     *
     * @param brightness - Brightness adjustment (-255 to 255)
     */
    setBrightness(brightness: number): void;
    /**
     * Apply the brightness filter to an ImageData object
     *
     * @param imageData - ImageData to filter
     * @returns Filtered ImageData
     */
    apply(imageData: ImageData): ImageData;
}
