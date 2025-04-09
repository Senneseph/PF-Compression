/**
 * Decoders for the PF-Compression PWA
 * 
 * This module provides decoder functions that transform encoded data
 * back into video frames. These are TypeScript equivalents
 * of the decode_* functions in webcam.py.
 */

import { checkerboard } from './utils';
import { EncodedData } from './encoders';

/**
 * Base class for decoders
 */
export abstract class Decoder {
  /**
   * Decode encoded data
   * 
   * @param encodedData - Data to decode
   * @returns Decoded ImageData
   */
  abstract decode(encodedData: EncodedData): ImageData;
}

/**
 * RGB Strobe decoder
 * 
 * Reconstructs an image by updating a persistent buffer with a single RGB channel.
 */
export class RGBStrobeDecoder extends Decoder {
  private persistentFrame: ImageData;
  
  /**
   * Create a new RGB Strobe decoder
   * 
   * @param width - Width of the frame
   * @param height - Height of the frame
   */
  constructor(width: number = 640, height: number = 480) {
    super();
    this.persistentFrame = new ImageData(width, height);
    
    // Initialize alpha channel to 255
    const data = this.persistentFrame.data;
    for (let i = 0; i < width * height; i++) {
      data[i * 4 + 3] = 255;
    }
  }
  
  /**
   * Reset the persistent frame
   */
  reset(): void {
    const { width, height } = this.persistentFrame;
    this.persistentFrame = new ImageData(width, height);
    
    // Initialize alpha channel to 255
    const data = this.persistentFrame.data;
    for (let i = 0; i < width * height; i++) {
      data[i * 4 + 3] = 255;
    }
  }
  
  /**
   * Decode encoded data by updating the persistent frame with a single RGB channel
   * 
   * @param encodedData - Encoded data with outputFrame and channelIndex
   * @returns Updated persistent frame
   */
  decode(encodedData: EncodedData): ImageData {
    const { outputFrame, channelIndex } = encodedData;
    
    // Extract the channel data from the encoded frame
    const { width, height, data: encodedData } = outputFrame;
    
    // Update the persistent frame
    const persistentData = this.persistentFrame.data;
    
    for (let i = 0; i < width * height; i++) {
      persistentData[i * 4 + channelIndex] = encodedData[i * 4 + channelIndex];
    }
    
    return this.persistentFrame;
  }
}

/**
 * RGB Even-Odd Strobe decoder
 * 
 * Reconstructs an image by updating a persistent buffer with even or odd values
 * from a single RGB channel.
 */
export class RGBEvenOddStrobeDecoder extends Decoder {
  private persistentFrame: ImageData;
  
  /**
   * Create a new RGB Even-Odd Strobe decoder
   * 
   * @param width - Width of the frame
   * @param height - Height of the frame
   */
  constructor(width: number = 640, height: number = 480) {
    super();
    this.persistentFrame = new ImageData(width, height);
    
    // Initialize alpha channel to 255
    const data = this.persistentFrame.data;
    for (let i = 0; i < width * height; i++) {
      data[i * 4 + 3] = 255;
    }
  }
  
  /**
   * Reset the persistent frame
   */
  reset(): void {
    const { width, height } = this.persistentFrame;
    this.persistentFrame = new ImageData(width, height);
    
    // Initialize alpha channel to 255
    const data = this.persistentFrame.data;
    for (let i = 0; i < width * height; i++) {
      data[i * 4 + 3] = 255;
    }
  }
  
  /**
   * Decode encoded data by updating the persistent frame with even or odd values
   * from a single RGB channel
   * 
   * @param encodedData - Encoded data with outputFrame, updateMask, channelIndex, and isOdd
   * @returns Updated persistent frame
   */
  decode(encodedData: EncodedData): ImageData {
    const { outputFrame, updateMask, channelIndex } = encodedData;
    
    // Extract the channel data from the encoded frame
    const { width, height, data: encodedData } = outputFrame;
    
    // Update the persistent frame
    const persistentData = this.persistentFrame.data;
    
    for (let i = 0; i < width * height; i++) {
      if (updateMask[i]) {
        persistentData[i * 4 + channelIndex] = encodedData[i * 4 + channelIndex];
      }
    }
    
    return this.persistentFrame;
  }
}

/**
 * RGB Matrix Strobe decoder
 * 
 * Reconstructs an image by updating a persistent buffer with pixels in a
 * checkerboard pattern from a single RGB channel.
 */
export class RGBMatrixStrobeDecoder extends Decoder {
  private persistentFrame: ImageData;
  
  /**
   * Create a new RGB Matrix Strobe decoder
   * 
   * @param width - Width of the frame
   * @param height - Height of the frame
   */
  constructor(width: number = 640, height: number = 480) {
    super();
    this.persistentFrame = new ImageData(width, height);
    
    // Initialize alpha channel to 255
    const data = this.persistentFrame.data;
    for (let i = 0; i < width * height; i++) {
      data[i * 4 + 3] = 255;
    }
  }
  
  /**
   * Reset the persistent frame
   */
  reset(): void {
    const { width, height } = this.persistentFrame;
    this.persistentFrame = new ImageData(width, height);
    
    // Initialize alpha channel to 255
    const data = this.persistentFrame.data;
    for (let i = 0; i < width * height; i++) {
      data[i * 4 + 3] = 255;
    }
  }
  
  /**
   * Decode encoded data by updating the persistent frame with pixels in a
   * checkerboard pattern from a single RGB channel
   * 
   * @param encodedData - Encoded data with outputFrame, channelIndex, and isOdd
   * @returns Updated persistent frame
   */
  decode(encodedData: EncodedData): ImageData {
    const { outputFrame, channelIndex, isOdd } = encodedData;
    
    // Extract the channel data from the encoded frame
    const { width, height, data: encodedData } = outputFrame;
    
    // Create checkerboard pattern
    const pattern = checkerboard(height, width);
    
    // Create update mask based on checkerboard pattern
    const updateMask = new Uint8Array(width * height);
    for (let i = 0; i < pattern.length; i++) {
      updateMask[i] = (pattern[i] === (isOdd ? 1 : 0)) ? 1 : 0;
    }
    
    // Update the persistent frame
    const persistentData = this.persistentFrame.data;
    
    for (let i = 0; i < width * height; i++) {
      if (updateMask[i]) {
        persistentData[i * 4 + channelIndex] = encodedData[i * 4 + channelIndex];
      }
    }
    
    return this.persistentFrame;
  }
}
