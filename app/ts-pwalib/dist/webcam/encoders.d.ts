/**
 * Encoders for the PF-Compression PWA
 *
 * This module provides encoder functions that transform video frames
 * into different representations. These are TypeScript equivalents
 * of the encode_* functions in webcam.py.
 */
/**
 * Interface for encoded data
 */
export interface EncodedData {
    [key: string]: any;
}
/**
 * Base class for encoders
 */
export declare abstract class Encoder {
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
export declare class RGBStrobeEncoder extends Encoder {
    private channelIndex;
    /**
     * Create a new RGB Strobe encoder
     *
     * @param channelIndex - Initial channel index (0=R, 1=G, 2=B)
     */
    constructor(channelIndex?: number);
    /**
     * Set the channel index
     *
     * @param channelIndex - Channel index (0=R, 1=G, 2=B)
     */
    setChannelIndex(channelIndex: number): void;
    /**
     * Get the next channel index in the cycle
     *
     * @returns Next channel index
     */
    nextChannelIndex(): number;
    /**
     * Encode an ImageData object by extracting a single RGB channel
     *
     * @param imageData - ImageData to encode
     * @returns Encoded data with the extracted channel and channel index
     */
    encode(imageData: ImageData): EncodedData;
}
/**
 * RGB Even-Odd Strobe encoder
 *
 * Extracts a single RGB channel and keeps only even or odd values.
 */
export declare class RGBEvenOddStrobeEncoder extends Encoder {
    private channelIndex;
    private isOdd;
    private cycleIndex;
    /**
     * Create a new RGB Even-Odd Strobe encoder
     *
     * @param channelIndex - Initial channel index (0=R, 1=G, 2=B)
     * @param isOdd - Whether to keep odd (true) or even (false) values
     */
    constructor(channelIndex?: number, isOdd?: boolean);
    /**
     * Set the channel index and parity
     *
     * @param channelIndex - Channel index (0=R, 1=G, 2=B)
     * @param isOdd - Whether to keep odd (true) or even (false) values
     */
    setParams(channelIndex: number, isOdd: boolean): void;
    /**
     * Get the next parameters in the cycle
     *
     * @returns [channelIndex, isOdd]
     */
    nextParams(): [number, boolean];
    /**
     * Encode an ImageData object by extracting a single RGB channel
     * and keeping only even or odd values
     *
     * @param imageData - ImageData to encode
     * @returns Encoded data with the extracted channel, update mask, channel index, and parity
     */
    encode(imageData: ImageData): EncodedData;
}
/**
 * RGB Matrix Strobe encoder
 *
 * Extracts a single RGB channel and keeps only pixels in a checkerboard pattern.
 */
export declare class RGBMatrixStrobeEncoder extends Encoder {
    private channelIndex;
    private isOdd;
    private cycleIndex;
    /**
     * Create a new RGB Matrix Strobe encoder
     *
     * @param channelIndex - Initial channel index (0=R, 1=G, 2=B)
     * @param isOdd - Whether to keep odd (true) or even (false) pixels
     */
    constructor(channelIndex?: number, isOdd?: boolean);
    /**
     * Set the channel index and parity
     *
     * @param channelIndex - Channel index (0=R, 1=G, 2=B)
     * @param isOdd - Whether to keep odd (true) or even (false) pixels
     */
    setParams(channelIndex: number, isOdd: boolean): void;
    /**
     * Get the next parameters in the cycle
     *
     * @returns [channelIndex, isOdd]
     */
    nextParams(): [number, boolean];
    /**
     * Encode an ImageData object by extracting a single RGB channel
     * and keeping only pixels in a checkerboard pattern
     *
     * @param imageData - ImageData to encode
     * @returns Encoded data with the extracted channel, channel index, and parity
     */
    encode(imageData: ImageData): EncodedData;
}
