/**
 * Utility functions for the PF-Compression PWA
 *
 * This module provides utility functions for numerical operations,
 * image processing, and other common tasks.
 */
/**
 * Create a new typed array filled with zeros
 *
 * @param shape - Shape of the array (e.g., [height, width, channels])
 * @param type - Type of the array (default: Uint8Array)
 * @returns Typed array filled with zeros
 */
export declare function zeros(shape: number[], type?: 'uint8' | 'uint16' | 'uint32' | 'int8' | 'int16' | 'int32' | 'float32' | 'float64'): TypedArray;
/**
 * Create a new typed array filled with ones
 *
 * @param shape - Shape of the array (e.g., [height, width, channels])
 * @param type - Type of the array (default: Uint8Array)
 * @returns Typed array filled with ones
 */
export declare function ones(shape: number[], type?: 'uint8' | 'uint16' | 'uint32' | 'int8' | 'int16' | 'int32' | 'float32' | 'float64'): TypedArray;
/**
 * Create a range of numbers
 *
 * @param start - Start value (inclusive)
 * @param stop - Stop value (exclusive)
 * @param step - Step size (default: 1)
 * @returns Array of numbers
 */
export declare function range(start: number, stop: number, step?: number): number[];
/**
 * Create a 2D array of indices
 *
 * @param height - Height of the array
 * @param width - Width of the array
 * @returns [row_indices, col_indices] arrays
 */
export declare function indices(height: number, width: number): [Uint32Array, Uint32Array];
/**
 * Create a checkerboard pattern
 *
 * @param height - Height of the pattern
 * @param width - Width of the pattern
 * @returns Uint8Array with checkerboard pattern (0 or 1)
 */
export declare function checkerboard(height: number, width: number): Uint8Array;
/**
 * Apply a bitwise operation to each element of an array
 *
 * @param array - Input array
 * @param operation - Bitwise operation to apply
 * @param operand - Operand for the operation
 * @returns New array with the operation applied
 */
export declare function bitwiseOp(array: TypedArray, operation: 'and' | 'or' | 'xor' | 'not' | 'shiftLeft' | 'shiftRight', operand?: number): TypedArray;
/**
 * Extract a channel from an ImageData object
 *
 * @param imageData - ImageData object
 * @param channel - Channel index (0=R, 1=G, 2=B, 3=A)
 * @returns Uint8Array with the channel data
 */
export declare function extractChannel(imageData: ImageData, channel: number): Uint8Array;
/**
 * Create an ImageData object from channel arrays
 *
 * @param width - Width of the image
 * @param height - Height of the image
 * @param r - Red channel data
 * @param g - Green channel data
 * @param b - Blue channel data
 * @param a - Alpha channel data (optional, defaults to 255)
 * @returns ImageData object
 */
export declare function createImageData(width: number, height: number, r: Uint8Array, g: Uint8Array, b: Uint8Array, a?: Uint8Array): ImageData;
/**
 * Find prime factors of a number
 *
 * @param n - Number to factorize
 * @returns Object with prime factors as keys and exponents as values
 */
export declare function primeFactors(n: number): Record<number, number>;
/**
 * Generate Fibonacci numbers up to a limit
 *
 * @param limit - Maximum value
 * @returns Array of Fibonacci numbers
 */
export declare function fibonacciSequence(limit: number): number[];
/**
 * Generate prime numbers up to a limit using the Sieve of Eratosthenes
 *
 * @param limit - Maximum value
 * @returns Array of prime numbers
 */
export declare function primeSequence(limit: number): number[];
type TypedArray = Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array;
export {};
