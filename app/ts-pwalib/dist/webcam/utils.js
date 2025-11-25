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
export function zeros(shape, type = 'uint8') {
    // Calculate total size
    const size = shape.reduce((a, b) => a * b, 1);
    // Create array of the specified type
    switch (type) {
        case 'uint8':
            return new Uint8Array(size);
        case 'uint16':
            return new Uint16Array(size);
        case 'uint32':
            return new Uint32Array(size);
        case 'int8':
            return new Int8Array(size);
        case 'int16':
            return new Int16Array(size);
        case 'int32':
            return new Int32Array(size);
        case 'float32':
            return new Float32Array(size);
        case 'float64':
            return new Float64Array(size);
        default:
            return new Uint8Array(size);
    }
}
/**
 * Create a new typed array filled with ones
 *
 * @param shape - Shape of the array (e.g., [height, width, channels])
 * @param type - Type of the array (default: Uint8Array)
 * @returns Typed array filled with ones
 */
export function ones(shape, type = 'uint8') {
    // Create array of the specified type
    const array = zeros(shape, type);
    // Fill with ones
    for (let i = 0; i < array.length; i++) {
        array[i] = 1;
    }
    return array;
}
/**
 * Create a range of numbers
 *
 * @param start - Start value (inclusive)
 * @param stop - Stop value (exclusive)
 * @param step - Step size (default: 1)
 * @returns Array of numbers
 */
export function range(start, stop, step = 1) {
    const result = [];
    for (let i = start; i < stop; i += step) {
        result.push(i);
    }
    return result;
}
/**
 * Create a 2D array of indices
 *
 * @param height - Height of the array
 * @param width - Width of the array
 * @returns [row_indices, col_indices] arrays
 */
export function indices(height, width) {
    const rowIndices = new Uint32Array(height * width);
    const colIndices = new Uint32Array(height * width);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = y * width + x;
            rowIndices[index] = y;
            colIndices[index] = x;
        }
    }
    return [rowIndices, colIndices];
}
/**
 * Create a checkerboard pattern
 *
 * @param height - Height of the pattern
 * @param width - Width of the pattern
 * @returns Uint8Array with checkerboard pattern (0 or 1)
 */
export function checkerboard(height, width) {
    const result = new Uint8Array(height * width);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = y * width + x;
            result[index] = (x + y) % 2;
        }
    }
    return result;
}
/**
 * Apply a bitwise operation to each element of an array
 *
 * @param array - Input array
 * @param operation - Bitwise operation to apply
 * @param operand - Operand for the operation
 * @returns New array with the operation applied
 */
export function bitwiseOp(array, operation, operand) {
    const result = new array.constructor(array.length);
    switch (operation) {
        case 'and':
            if (operand === undefined)
                throw new Error('Operand required for bitwise AND');
            for (let i = 0; i < array.length; i++) {
                result[i] = array[i] & operand;
            }
            break;
        case 'or':
            if (operand === undefined)
                throw new Error('Operand required for bitwise OR');
            for (let i = 0; i < array.length; i++) {
                result[i] = array[i] | operand;
            }
            break;
        case 'xor':
            if (operand === undefined)
                throw new Error('Operand required for bitwise XOR');
            for (let i = 0; i < array.length; i++) {
                result[i] = array[i] ^ operand;
            }
            break;
        case 'not':
            for (let i = 0; i < array.length; i++) {
                result[i] = ~array[i];
            }
            break;
        case 'shiftLeft':
            if (operand === undefined)
                throw new Error('Operand required for left shift');
            for (let i = 0; i < array.length; i++) {
                result[i] = array[i] << operand;
            }
            break;
        case 'shiftRight':
            if (operand === undefined)
                throw new Error('Operand required for right shift');
            for (let i = 0; i < array.length; i++) {
                result[i] = array[i] >> operand;
            }
            break;
        default:
            throw new Error(`Unknown bitwise operation: ${operation}`);
    }
    return result;
}
/**
 * Extract a channel from an ImageData object
 *
 * @param imageData - ImageData object
 * @param channel - Channel index (0=R, 1=G, 2=B, 3=A)
 * @returns Uint8Array with the channel data
 */
export function extractChannel(imageData, channel) {
    const { width, height, data } = imageData;
    const result = new Uint8Array(width * height);
    for (let i = 0; i < width * height; i++) {
        result[i] = data[i * 4 + channel];
    }
    return result;
}
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
export function createImageData(width, height, r, g, b, a) {
    const data = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i++) {
        data[i * 4] = r[i];
        data[i * 4 + 1] = g[i];
        data[i * 4 + 2] = b[i];
        data[i * 4 + 3] = a ? a[i] : 255;
    }
    return new ImageData(data, width, height);
}
/**
 * Find prime factors of a number
 *
 * @param n - Number to factorize
 * @returns Object with prime factors as keys and exponents as values
 */
export function primeFactors(n) {
    const factors = {};
    // Handle 2 separately
    while (n % 2 === 0) {
        factors[2] = (factors[2] || 0) + 1;
        n /= 2;
    }
    // Check odd numbers up to sqrt(n)
    for (let i = 3; i <= Math.sqrt(n); i += 2) {
        while (n % i === 0) {
            factors[i] = (factors[i] || 0) + 1;
            n /= i;
        }
    }
    // If n is a prime number greater than 2
    if (n > 2) {
        factors[n] = (factors[n] || 0) + 1;
    }
    return factors;
}
/**
 * Generate Fibonacci numbers up to a limit
 *
 * @param limit - Maximum value
 * @returns Array of Fibonacci numbers
 */
export function fibonacciSequence(limit) {
    const sequence = [0, 1];
    while (sequence[sequence.length - 1] + sequence[sequence.length - 2] <= limit) {
        sequence.push(sequence[sequence.length - 1] + sequence[sequence.length - 2]);
    }
    return sequence;
}
/**
 * Generate prime numbers up to a limit using the Sieve of Eratosthenes
 *
 * @param limit - Maximum value
 * @returns Array of prime numbers
 */
export function primeSequence(limit) {
    // Create a boolean array for the sieve
    const sieve = new Array(limit + 1).fill(true);
    sieve[0] = sieve[1] = false;
    // Mark non-prime numbers
    for (let i = 2; i * i <= limit; i++) {
        if (sieve[i]) {
            for (let j = i * i; j <= limit; j += i) {
                sieve[j] = false;
            }
        }
    }
    // Collect prime numbers
    const primes = [];
    for (let i = 2; i <= limit; i++) {
        if (sieve[i]) {
            primes.push(i);
        }
    }
    return primes;
}
