import { Effect } from '../core/effect';
/**
 * Effect that transforms RGB values to the nearest Fibonacci numbers.
 *
 * This effect:
 * 1. Computes the nearest Fibonacci number for each RGB component
 * 2. Preserves the alpha channel
 */
export declare class FibonacciRGBEffect extends Effect {
    /**
     * Lookup table for nearest Fibonacci numbers (0-255).
     */
    private readonly fibonacciLookup;
    /**
     * Initialize the Fibonacci RGB Effect.
     */
    constructor();
    /**
     * Transform an ImageData object by converting RGB values to the nearest Fibonacci numbers.
     *
     * @param imageData - The ImageData object to transform.
     * @returns The transformed ImageData object.
     */
    transform(imageData: ImageData): ImageData;
    /**
     * Precompute a lookup table for the nearest Fibonacci number for values from 0 to max_value.
     *
     * @param maxValue - Maximum value to precompute.
     * @returns Lookup table of nearest Fibonacci numbers.
     */
    private precomputeNearestFibonacci;
}
