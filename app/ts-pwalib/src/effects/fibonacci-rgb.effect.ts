import { Effect } from '../core/effect';

/**
 * Effect that transforms RGB values to the nearest Fibonacci numbers.
 * 
 * This effect:
 * 1. Computes the nearest Fibonacci number for each RGB component
 * 2. Preserves the alpha channel
 */
export class FibonacciRGBEffect extends Effect {
  /**
   * Lookup table for nearest Fibonacci numbers (0-255).
   */
  private readonly fibonacciLookup: Uint8Array;

  /**
   * Initialize the Fibonacci RGB Effect.
   */
  constructor() {
    super(
      'Fibonacci RGB',
      'Transforms RGB values to the nearest Fibonacci numbers, creating a distinctive color shift.'
    );
    
    // Initialize the Fibonacci number lookup table
    this.fibonacciLookup = this.precomputeNearestFibonacci(255);
  }

  /**
   * Transform an ImageData object by converting RGB values to the nearest Fibonacci numbers.
   * 
   * @param imageData - The ImageData object to transform.
   * @returns The transformed ImageData object.
   */
  transform(imageData: ImageData): ImageData {
    // Validate the input
    this.validateImageData(imageData);

    // Create a new ImageData object for the output
    const output = new ImageData(imageData.width, imageData.height);
    
    // Get the data arrays
    const inputData = imageData.data;
    const outputData = output.data;
    
    // Process each pixel
    for (let i = 0; i < inputData.length; i += 4) {
      // Convert each RGB component to the nearest Fibonacci number
      outputData[i] = this.fibonacciLookup[inputData[i]];         // R
      outputData[i + 1] = this.fibonacciLookup[inputData[i + 1]]; // G
      outputData[i + 2] = this.fibonacciLookup[inputData[i + 2]]; // B
      
      // Preserve the alpha channel
      outputData[i + 3] = inputData[i + 3];                       // A
    }
    
    return output;
  }

  /**
   * Precompute a lookup table for the nearest Fibonacci number for values from 0 to max_value.
   * 
   * @param maxValue - Maximum value to precompute.
   * @returns Lookup table of nearest Fibonacci numbers.
   */
  private precomputeNearestFibonacci(maxValue: number): Uint8Array {
    // Generate Fibonacci numbers up to maxValue
    const fibs: number[] = [0, 1];
    while (fibs[fibs.length - 1] + fibs[fibs.length - 2] <= maxValue) {
      fibs.push(fibs[fibs.length - 1] + fibs[fibs.length - 2]);
    }
    
    // Create lookup table
    const lookup = new Uint8Array(maxValue + 1);
    
    // For each value, find the nearest Fibonacci number
    for (let i = 0; i <= maxValue; i++) {
      // Find the nearest Fibonacci number
      let nearestFib = fibs[0];
      let minDistance = Math.abs(nearestFib - i);
      
      for (let j = 1; j < fibs.length; j++) {
        const distance = Math.abs(fibs[j] - i);
        if (distance < minDistance) {
          minDistance = distance;
          nearestFib = fibs[j];
        }
      }
      
      lookup[i] = nearestFib;
    }
    
    return lookup;
  }
}
