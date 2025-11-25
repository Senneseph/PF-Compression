/**
 * Transformers for the PF-Compression PWA
 *
 * This module provides transformer functions that apply effects to video frames.
 * These are TypeScript equivalents of the transformer_* functions in webcam.py.
 */
import { WebGLRenderer } from './webgl-renderer';
/**
 * Base class for WebGL-based transformers
 */
export declare abstract class WebGLTransformer {
    protected renderer: WebGLRenderer;
    protected fragmentShader: string;
    /**
     * Create a new WebGL transformer
     */
    constructor();
    /**
     * Get the fragment shader source code
     */
    protected abstract getFragmentShader(): string;
    /**
     * Transform an ImageData object
     *
     * @param imageData - ImageData to transform
     * @returns Transformed ImageData
     */
    transform(imageData: ImageData): ImageData;
    /**
     * Clean up resources
     */
    dispose(): void;
}
/**
 * Color Negative transformer
 *
 * Inverts the colors of the image.
 */
export declare class ColorNegativeTransformer extends WebGLTransformer {
    protected getFragmentShader(): string;
}
/**
 * Prime RGB transformer
 *
 * Maps each RGB component to the nearest prime number.
 */
export declare class PrimeRGBTransformer extends WebGLTransformer {
    private primeLookup;
    constructor();
    protected getFragmentShader(): string;
    transform(imageData: ImageData): ImageData;
}
/**
 * Fibonacci RGB transformer
 *
 * Maps each RGB component to the nearest Fibonacci number.
 */
export declare class FibonacciRGBTransformer extends WebGLTransformer {
    private fibLookup;
    constructor();
    protected getFragmentShader(): string;
    transform(imageData: ImageData): ImageData;
}
/**
 * Middle Four Bits transformer
 *
 * Extracts the middle 4 bits from each RGB component.
 */
export declare class MiddleFourBitsTransformer extends WebGLTransformer {
    protected getFragmentShader(): string;
}
