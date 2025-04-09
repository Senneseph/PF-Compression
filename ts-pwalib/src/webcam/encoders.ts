/**
 * Encoders for the PF-Compression PWA
 * 
 * This module provides encoder functions that transform video frames
 * into different representations. These are TypeScript equivalents
 * of the encode_* functions in webcam.py.
 */

import { extractChannel, checkerboard } from './utils';

/**
 * Interface for encoded data
 */
export interface EncodedData {
  [key: string]: any;
}

/**
 * Base class for encoders
 */
export abstract class Encoder {
  /**
   * Encode an ImageData object
   * 
   * @param imageData - ImageData to encode
   * @returns Encoded data
   */
  abstract encode(imageData: ImageData): EncodedData;
}

/**
 * RGB Strobe encoder
 * 
 * Extracts a single RGB channel.
 */
export class RGBStrobeEncoder extends Encoder {
  private channelIndex: number = 0;
  
  /**
   * Create a new RGB Strobe encoder
   * 
   * @param channelIndex - Initial channel index (0=R, 1=G, 2=B)
   */
  constructor(channelIndex: number = 0) {
    super();
    this.channelIndex = channelIndex % 3;
  }
  
  /**
   * Set the channel index
   * 
   * @param channelIndex - Channel index (0=R, 1=G, 2=B)
   */
  setChannelIndex(channelIndex: number): void {
    this.channelIndex = channelIndex % 3;
  }
  
  /**
   * Get the next channel index in the cycle
   * 
   * @returns Next channel index
   */
  nextChannelIndex(): number {
    this.channelIndex = (this.channelIndex + 1) % 3;
    return this.channelIndex;
  }
  
  /**
   * Encode an ImageData object by extracting a single RGB channel
   * 
   * @param imageData - ImageData to encode
   * @returns Encoded data with the extracted channel and channel index
   */
  encode(imageData: ImageData): EncodedData {
    const { width, height } = imageData;
    
    // Create output image with only the selected channel
    const output = new ImageData(width, height);
    const outputData = output.data;
    
    // Extract the channel data
    const channelData = extractChannel(imageData, this.channelIndex);
    
    // Set the selected channel in the output
    for (let i = 0; i < width * height; i++) {
      outputData[i * 4 + this.channelIndex] = channelData[i];
      outputData[i * 4 + 3] = 255; // Alpha
    }
    
    return {
      outputFrame: output,
      channelIndex: this.channelIndex
    };
  }
}

/**
 * RGB Even-Odd Strobe encoder
 * 
 * Extracts a single RGB channel and keeps only even or odd values.
 */
export class RGBEvenOddStrobeEncoder extends Encoder {
  private channelIndex: number = 0;
  private isOdd: boolean = true;
  private cycleIndex: number = 0;
  
  /**
   * Create a new RGB Even-Odd Strobe encoder
   * 
   * @param channelIndex - Initial channel index (0=R, 1=G, 2=B)
   * @param isOdd - Whether to keep odd (true) or even (false) values
   */
  constructor(channelIndex: number = 0, isOdd: boolean = true) {
    super();
    this.channelIndex = channelIndex % 3;
    this.isOdd = isOdd;
  }
  
  /**
   * Set the channel index and parity
   * 
   * @param channelIndex - Channel index (0=R, 1=G, 2=B)
   * @param isOdd - Whether to keep odd (true) or even (false) values
   */
  setParams(channelIndex: number, isOdd: boolean): void {
    this.channelIndex = channelIndex % 3;
    this.isOdd = isOdd;
  }
  
  /**
   * Get the next parameters in the cycle
   * 
   * @returns [channelIndex, isOdd]
   */
  nextParams(): [number, boolean] {
    // Cycle through all combinations of channel and parity
    const cycleOrder: [number, boolean][] = [
      [0, true],  // R odd
      [1, false], // G even
      [2, true],  // B odd
      [0, false], // R even
      [1, true],  // G odd
      [2, false]  // B even
    ];
    
    this.cycleIndex = (this.cycleIndex + 1) % cycleOrder.length;
    [this.channelIndex, this.isOdd] = cycleOrder[this.cycleIndex];
    
    return [this.channelIndex, this.isOdd];
  }
  
  /**
   * Encode an ImageData object by extracting a single RGB channel
   * and keeping only even or odd values
   * 
   * @param imageData - ImageData to encode
   * @returns Encoded data with the extracted channel, update mask, channel index, and parity
   */
  encode(imageData: ImageData): EncodedData {
    const { width, height } = imageData;
    
    // Create output image
    const output = new ImageData(width, height);
    const outputData = output.data;
    
    // Extract the channel data
    const channelData = extractChannel(imageData, this.channelIndex);
    
    // Create update mask based on parity
    const updateMask = new Uint8Array(width * height);
    for (let i = 0; i < channelData.length; i++) {
      updateMask[i] = (channelData[i] % 2 === (this.isOdd ? 1 : 0)) ? 1 : 0;
    }
    
    // Apply the update mask to keep only matching values
    for (let i = 0; i < width * height; i++) {
      if (updateMask[i]) {
        outputData[i * 4 + this.channelIndex] = channelData[i];
      }
      outputData[i * 4 + 3] = 255; // Alpha
    }
    
    return {
      outputFrame: output,
      updateMask,
      channelIndex: this.channelIndex,
      isOdd: this.isOdd
    };
  }
}

/**
 * RGB Matrix Strobe encoder
 * 
 * Extracts a single RGB channel and keeps only pixels in a checkerboard pattern.
 */
export class RGBMatrixStrobeEncoder extends Encoder {
  private channelIndex: number = 0;
  private isOdd: boolean = true;
  private cycleIndex: number = 0;
  
  /**
   * Create a new RGB Matrix Strobe encoder
   * 
   * @param channelIndex - Initial channel index (0=R, 1=G, 2=B)
   * @param isOdd - Whether to keep odd (true) or even (false) pixels
   */
  constructor(channelIndex: number = 0, isOdd: boolean = true) {
    super();
    this.channelIndex = channelIndex % 3;
    this.isOdd = isOdd;
  }
  
  /**
   * Set the channel index and parity
   * 
   * @param channelIndex - Channel index (0=R, 1=G, 2=B)
   * @param isOdd - Whether to keep odd (true) or even (false) pixels
   */
  setParams(channelIndex: number, isOdd: boolean): void {
    this.channelIndex = channelIndex % 3;
    this.isOdd = isOdd;
  }
  
  /**
   * Get the next parameters in the cycle
   * 
   * @returns [channelIndex, isOdd]
   */
  nextParams(): [number, boolean] {
    // Cycle through all combinations of channel and parity
    const cycleOrder: [number, boolean][] = [
      [0, true],  // R odd
      [1, false], // G even
      [2, true],  // B odd
      [0, false], // R even
      [1, true],  // G odd
      [2, false]  // B even
    ];
    
    this.cycleIndex = (this.cycleIndex + 1) % cycleOrder.length;
    [this.channelIndex, this.isOdd] = cycleOrder[this.cycleIndex];
    
    return [this.channelIndex, this.isOdd];
  }
  
  /**
   * Encode an ImageData object by extracting a single RGB channel
   * and keeping only pixels in a checkerboard pattern
   * 
   * @param imageData - ImageData to encode
   * @returns Encoded data with the extracted channel, channel index, and parity
   */
  encode(imageData: ImageData): EncodedData {
    const { width, height } = imageData;
    
    // Create output image
    const output = new ImageData(width, height);
    const outputData = output.data;
    
    // Extract the channel data
    const channelData = extractChannel(imageData, this.channelIndex);
    
    // Create checkerboard pattern
    const pattern = checkerboard(height, width);
    
    // Create update mask based on checkerboard pattern
    const updateMask = new Uint8Array(width * height);
    for (let i = 0; i < pattern.length; i++) {
      updateMask[i] = (pattern[i] === (this.isOdd ? 1 : 0)) ? 1 : 0;
    }
    
    // Apply the update mask to keep only matching pixels
    for (let i = 0; i < width * height; i++) {
      if (updateMask[i]) {
        outputData[i * 4 + this.channelIndex] = channelData[i];
      }
      outputData[i * 4 + 3] = 255; // Alpha
    }
    
    return {
      outputFrame: output,
      channelIndex: this.channelIndex,
      isOdd: this.isOdd
    };
  }
}
