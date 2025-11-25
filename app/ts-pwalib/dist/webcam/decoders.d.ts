/**
 * Decoders for the PF-Compression PWA
 *
 * This module provides decoder functions that transform encoded data
 * back into video frames. These are TypeScript equivalents
 * of the decode_* functions in webcam.py.
 */
import { EncodedData } from './encoders';
/**
 * Base class for decoders
 */
export declare abstract class Decoder {
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
export declare class RGBStrobeDecoder extends Decoder {
    private persistentFrame;
    /**
     * Create a new RGB Strobe decoder
     *
     * @param width - Width of the frame
     * @param height - Height of the frame
     */
    constructor(width?: number, height?: number);
    /**
     * Reset the persistent frame
     */
    reset(): void;
    /**
     * Decode encoded data by updating the persistent frame with a single RGB channel
     *
     * @param encodedData - Encoded data with outputFrame and channelIndex
     * @returns Updated persistent frame
     */
    decode(encodedData: EncodedData): ImageData;
}
/**
 * RGB Even-Odd Strobe decoder
 *
 * Reconstructs an image by updating a persistent buffer with even or odd values
 * from a single RGB channel.
 */
export declare class RGBEvenOddStrobeDecoder extends Decoder {
    private persistentFrame;
    /**
     * Create a new RGB Even-Odd Strobe decoder
     *
     * @param width - Width of the frame
     * @param height - Height of the frame
     */
    constructor(width?: number, height?: number);
    /**
     * Reset the persistent frame
     */
    reset(): void;
    /**
     * Decode encoded data by updating the persistent frame with even or odd values
     * from a single RGB channel
     *
     * @param encodedData - Encoded data with outputFrame, updateMask, channelIndex, and isOdd
     * @returns Updated persistent frame
     */
    decode(encodedData: EncodedData): ImageData;
}
/**
 * RGB Matrix Strobe decoder
 *
 * Reconstructs an image by updating a persistent buffer with pixels in a
 * checkerboard pattern from a single RGB channel.
 */
export declare class RGBMatrixStrobeDecoder extends Decoder {
    private persistentFrame;
    /**
     * Create a new RGB Matrix Strobe decoder
     *
     * @param width - Width of the frame
     * @param height - Height of the frame
     */
    constructor(width?: number, height?: number);
    /**
     * Reset the persistent frame
     */
    reset(): void;
    /**
     * Decode encoded data by updating the persistent frame with pixels in a
     * checkerboard pattern from a single RGB channel
     *
     * @param encodedData - Encoded data with outputFrame, channelIndex, and isOdd
     * @returns Updated persistent frame
     */
    decode(encodedData: EncodedData): ImageData;
}
