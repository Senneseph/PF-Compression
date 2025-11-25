/**
 * Transformers for the PF-Compression PWA
 *
 * This module provides transformer functions that apply effects to video frames.
 * These are TypeScript equivalents of the transformer_* functions in webcam.py.
 */
import { WebGLRenderer } from './webgl-renderer';
import { primeSequence, fibonacciSequence } from './utils';
/**
 * Base class for WebGL-based transformers
 */
export class WebGLTransformer {
    /**
     * Create a new WebGL transformer
     */
    constructor() {
        this.renderer = new WebGLRenderer();
        this.fragmentShader = this.getFragmentShader();
        this.renderer.setFragmentShader(this.fragmentShader);
    }
    /**
     * Transform an ImageData object
     *
     * @param imageData - ImageData to transform
     * @returns Transformed ImageData
     */
    transform(imageData) {
        return this.renderer.render(imageData);
    }
    /**
     * Clean up resources
     */
    dispose() {
        this.renderer.dispose();
    }
}
/**
 * Color Negative transformer
 *
 * Inverts the colors of the image.
 */
export class ColorNegativeTransformer extends WebGLTransformer {
    getFragmentShader() {
        return `
      precision mediump float;
      uniform sampler2D u_image;
      varying vec2 v_texCoord;
      
      void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        gl_FragColor = vec4(1.0 - color.rgb, color.a);
      }
    `;
    }
}
/**
 * Prime RGB transformer
 *
 * Maps each RGB component to the nearest prime number.
 */
export class PrimeRGBTransformer extends WebGLTransformer {
    constructor() {
        super();
        this.primeLookup = [];
        // Generate prime lookup table
        const primes = primeSequence(255);
        this.primeLookup = new Array(256).fill(0);
        for (let i = 0; i <= 255; i++) {
            let nearestPrime = primes[0];
            let minDistance = Math.abs(nearestPrime - i);
            for (let j = 1; j < primes.length; j++) {
                const distance = Math.abs(primes[j] - i);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestPrime = primes[j];
                }
            }
            this.primeLookup[i] = nearestPrime;
        }
    }
    getFragmentShader() {
        // Create a texture with the prime lookup table
        const lookupTexture = new Uint8Array(256 * 4);
        for (let i = 0; i < 256; i++) {
            lookupTexture[i * 4] = this.primeLookup[i];
            lookupTexture[i * 4 + 1] = this.primeLookup[i];
            lookupTexture[i * 4 + 2] = this.primeLookup[i];
            lookupTexture[i * 4 + 3] = 255;
        }
        return `
      precision mediump float;
      uniform sampler2D u_image;
      uniform sampler2D u_lookup;
      varying vec2 v_texCoord;
      
      void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        
        // Look up nearest prime for each channel
        float r = texture2D(u_lookup, vec2(color.r, 0.5)).r;
        float g = texture2D(u_lookup, vec2(color.g, 0.5)).g;
        float b = texture2D(u_lookup, vec2(color.b, 0.5)).b;
        
        gl_FragColor = vec4(r, g, b, color.a);
      }
    `;
    }
    transform(imageData) {
        // For WebGL transformers that need lookup tables,
        // we'll need to implement a CPU-based fallback
        // since WebGL doesn't support dynamic texture lookups easily
        // Create output image
        const output = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
        // Apply prime mapping
        for (let i = 0; i < output.data.length; i += 4) {
            output.data[i] = this.primeLookup[output.data[i]];
            output.data[i + 1] = this.primeLookup[output.data[i + 1]];
            output.data[i + 2] = this.primeLookup[output.data[i + 2]];
        }
        return output;
    }
}
/**
 * Fibonacci RGB transformer
 *
 * Maps each RGB component to the nearest Fibonacci number.
 */
export class FibonacciRGBTransformer extends WebGLTransformer {
    constructor() {
        super();
        this.fibLookup = [];
        // Generate Fibonacci lookup table
        const fibs = fibonacciSequence(255);
        this.fibLookup = new Array(256).fill(0);
        for (let i = 0; i <= 255; i++) {
            let nearestFib = fibs[0];
            let minDistance = Math.abs(nearestFib - i);
            for (let j = 1; j < fibs.length; j++) {
                const distance = Math.abs(fibs[j] - i);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestFib = fibs[j];
                }
            }
            this.fibLookup[i] = nearestFib;
        }
    }
    getFragmentShader() {
        return `
      precision mediump float;
      uniform sampler2D u_image;
      varying vec2 v_texCoord;
      
      void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        gl_FragColor = color; // Placeholder - actual implementation uses lookup table
      }
    `;
    }
    transform(imageData) {
        // Create output image
        const output = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
        // Apply Fibonacci mapping
        for (let i = 0; i < output.data.length; i += 4) {
            output.data[i] = this.fibLookup[output.data[i]];
            output.data[i + 1] = this.fibLookup[output.data[i + 1]];
            output.data[i + 2] = this.fibLookup[output.data[i + 2]];
        }
        return output;
    }
}
/**
 * Middle Four Bits transformer
 *
 * Extracts the middle 4 bits from each RGB component.
 */
export class MiddleFourBitsTransformer extends WebGLTransformer {
    getFragmentShader() {
        return `
      precision mediump float;
      uniform sampler2D u_image;
      varying vec2 v_texCoord;
      
      void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        
        // Extract middle 4 bits (bits 2-5) and scale to 0-255
        vec3 middleBits = floor(color.rgb * 255.0);
        middleBits = floor(mod(middleBits / 4.0, 16.0)) * 16.0;
        
        gl_FragColor = vec4(middleBits / 255.0, color.a);
      }
    `;
    }
}
