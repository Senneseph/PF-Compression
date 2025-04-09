/**
 * Filters for the PF-Compression PWA
 * 
 * This module provides filter functions that modify video frames.
 * These are TypeScript equivalents of the filter_* functions in webcam.py.
 */

/**
 * Base class for filters
 */
export abstract class Filter {
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
export class ColorNegativeFilter extends Filter {
  /**
   * Apply the color negative filter to an ImageData object
   * 
   * @param imageData - ImageData to filter
   * @returns Filtered ImageData
   */
  apply(imageData: ImageData): ImageData {
    const { width, height, data } = imageData;
    
    // Create output image
    const output = new ImageData(width, height);
    const outputData = output.data;
    
    // Invert colors
    for (let i = 0; i < data.length; i += 4) {
      outputData[i] = 255 - data[i];         // R
      outputData[i + 1] = 255 - data[i + 1]; // G
      outputData[i + 2] = 255 - data[i + 2]; // B
      outputData[i + 3] = data[i + 3];       // A
    }
    
    return output;
  }
}

/**
 * Middle Four Bits filter
 * 
 * Extracts the middle 4 bits from each RGB component.
 */
export class MiddleFourBitsFilter extends Filter {
  /**
   * Apply the middle four bits filter to an ImageData object
   * 
   * @param imageData - ImageData to filter
   * @returns Filtered ImageData
   */
  apply(imageData: ImageData): ImageData {
    const { width, height, data } = imageData;
    
    // Create output image
    const output = new ImageData(width, height);
    const outputData = output.data;
    
    // Extract middle 4 bits
    for (let i = 0; i < data.length; i += 4) {
      // Extract bits 2-5 (0-indexed) and shift to bits 0-3
      // Then scale to 0-255 range by multiplying by 16
      outputData[i] = ((data[i] >> 2) & 0x0F) * 16;       // R
      outputData[i + 1] = ((data[i + 1] >> 2) & 0x0F) * 16; // G
      outputData[i + 2] = ((data[i + 2] >> 2) & 0x0F) * 16; // B
      outputData[i + 3] = data[i + 3];                     // A
    }
    
    return output;
  }
}

/**
 * Grayscale filter
 * 
 * Converts the image to grayscale.
 */
export class GrayscaleFilter extends Filter {
  /**
   * Apply the grayscale filter to an ImageData object
   * 
   * @param imageData - ImageData to filter
   * @returns Filtered ImageData
   */
  apply(imageData: ImageData): ImageData {
    const { width, height, data } = imageData;
    
    // Create output image
    const output = new ImageData(width, height);
    const outputData = output.data;
    
    // Convert to grayscale
    for (let i = 0; i < data.length; i += 4) {
      // Use luminance formula: 0.299*R + 0.587*G + 0.114*B
      const gray = Math.round(
        0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
      );
      
      outputData[i] = gray;     // R
      outputData[i + 1] = gray; // G
      outputData[i + 2] = gray; // B
      outputData[i + 3] = data[i + 3]; // A
    }
    
    return output;
  }
}

/**
 * Threshold filter
 * 
 * Applies a threshold to the image, converting it to black and white.
 */
export class ThresholdFilter extends Filter {
  private threshold: number;
  
  /**
   * Create a new Threshold filter
   * 
   * @param threshold - Threshold value (0-255)
   */
  constructor(threshold: number = 128) {
    super();
    this.threshold = Math.max(0, Math.min(255, threshold));
  }
  
  /**
   * Set the threshold value
   * 
   * @param threshold - Threshold value (0-255)
   */
  setThreshold(threshold: number): void {
    this.threshold = Math.max(0, Math.min(255, threshold));
  }
  
  /**
   * Apply the threshold filter to an ImageData object
   * 
   * @param imageData - ImageData to filter
   * @returns Filtered ImageData
   */
  apply(imageData: ImageData): ImageData {
    const { width, height, data } = imageData;
    
    // Create output image
    const output = new ImageData(width, height);
    const outputData = output.data;
    
    // Apply threshold
    for (let i = 0; i < data.length; i += 4) {
      // Calculate grayscale value
      const gray = Math.round(
        0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
      );
      
      // Apply threshold
      const value = gray >= this.threshold ? 255 : 0;
      
      outputData[i] = value;     // R
      outputData[i + 1] = value; // G
      outputData[i + 2] = value; // B
      outputData[i + 3] = data[i + 3]; // A
    }
    
    return output;
  }
}

/**
 * Brightness filter
 * 
 * Adjusts the brightness of the image.
 */
export class BrightnessFilter extends Filter {
  private brightness: number;
  
  /**
   * Create a new Brightness filter
   * 
   * @param brightness - Brightness adjustment (-255 to 255)
   */
  constructor(brightness: number = 0) {
    super();
    this.brightness = Math.max(-255, Math.min(255, brightness));
  }
  
  /**
   * Set the brightness value
   * 
   * @param brightness - Brightness adjustment (-255 to 255)
   */
  setBrightness(brightness: number): void {
    this.brightness = Math.max(-255, Math.min(255, brightness));
  }
  
  /**
   * Apply the brightness filter to an ImageData object
   * 
   * @param imageData - ImageData to filter
   * @returns Filtered ImageData
   */
  apply(imageData: ImageData): ImageData {
    const { width, height, data } = imageData;
    
    // Create output image
    const output = new ImageData(width, height);
    const outputData = output.data;
    
    // Apply brightness adjustment
    for (let i = 0; i < data.length; i += 4) {
      outputData[i] = Math.max(0, Math.min(255, data[i] + this.brightness));         // R
      outputData[i + 1] = Math.max(0, Math.min(255, data[i + 1] + this.brightness)); // G
      outputData[i + 2] = Math.max(0, Math.min(255, data[i + 2] + this.brightness)); // B
      outputData[i + 3] = data[i + 3]; // A
    }
    
    return output;
  }
}
