import { Effect } from '../core/effect';
/**
 * Effect that transforms RGB values to the nearest prime numbers.
 *
 * This effect:
 * 1. Computes the nearest prime number for each RGB component
 * 2. Preserves the alpha channel
 */
export declare class PrimeRGBEffect extends Effect {
    /**
     * Lookup table for nearest prime numbers (0-255).
     */
    private readonly primeLookup;
    /**
     * Initialize the Prime RGB Effect.
     */
    constructor();
    /**
     * Transform an ImageData object by converting RGB values to the nearest prime numbers.
     *
     * @param imageData - The ImageData object to transform.
     * @returns The transformed ImageData object.
     */
    transform(imageData: ImageData): ImageData;
    /**
     * Precompute a lookup table for the nearest prime number for values from 0 to max_value.
     *
     * @param maxValue - Maximum value to precompute.
     * @returns Lookup table of nearest prime numbers.
     */
    private precomputeNearestPrimes;
}
