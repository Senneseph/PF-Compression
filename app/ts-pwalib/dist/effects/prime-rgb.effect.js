import { Effect } from '../core/effect';
/**
 * Effect that transforms RGB values to the nearest prime numbers.
 *
 * This effect:
 * 1. Computes the nearest prime number for each RGB component
 * 2. Preserves the alpha channel
 */
export class PrimeRGBEffect extends Effect {
    /**
     * Initialize the Prime RGB Effect.
     */
    constructor() {
        super('Prime RGB', 'Transforms RGB values to the nearest prime numbers, creating a distinctive color shift.');
        // Initialize the prime number lookup table
        this.primeLookup = this.precomputeNearestPrimes(255);
    }
    /**
     * Transform an ImageData object by converting RGB values to the nearest prime numbers.
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
            // Convert each RGB component to the nearest prime number
            outputData[i] = this.primeLookup[inputData[i]]; // R
            outputData[i + 1] = this.primeLookup[inputData[i + 1]]; // G
            outputData[i + 2] = this.primeLookup[inputData[i + 2]]; // B
            // Preserve the alpha channel
            outputData[i + 3] = inputData[i + 3]; // A
        }
        return output;
    }
    /**
     * Precompute a lookup table for the nearest prime number for values from 0 to max_value.
     *
     * @param maxValue - Maximum value to precompute.
     * @returns Lookup table of nearest prime numbers.
     */
    precomputeNearestPrimes(maxValue) {
        // List of primes up to 255
        const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
            157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
            239, 241, 251];
        // Create lookup table
        const lookup = new Uint8Array(maxValue + 1);
        // For each value, find the nearest prime
        for (let i = 0; i <= maxValue; i++) {
            // Find the nearest prime
            let nearestPrime = primes[0];
            let minDistance = Math.abs(nearestPrime - i);
            for (let j = 1; j < primes.length; j++) {
                const distance = Math.abs(primes[j] - i);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestPrime = primes[j];
                }
            }
            lookup[i] = nearestPrime;
        }
        return lookup;
    }
}
