/**
 * WebGL Renderer for GPU-accelerated video processing
 *
 * This module provides a WebGL-based renderer for applying effects to video frames.
 * It handles shader compilation, texture management, and rendering.
 */
/**
 * WebGL Renderer class for GPU-accelerated video processing
 */
export declare class WebGLRenderer {
    private gl;
    private canvas;
    private program;
    private positionBuffer;
    private texCoordBuffer;
    private texture;
    private framebuffer;
    private outputTexture;
    private width;
    private height;
    /**
     * Create a new WebGL Renderer
     *
     * @param canvas - Canvas element to render to, or create a new one if not provided
     */
    constructor(canvas?: HTMLCanvasElement);
    /**
     * Initialize WebGL context and resources
     */
    private initWebGL;
    /**
     * Create a WebGL shader
     *
     * @param type - Shader type (VERTEX_SHADER or FRAGMENT_SHADER)
     * @param source - GLSL source code
     * @returns Compiled shader
     */
    private createShader;
    /**
     * Create a WebGL program from shaders
     *
     * @param vertexShader - Vertex shader
     * @param fragmentShader - Fragment shader
     * @returns Linked program
     */
    private createProgram;
    /**
     * Set the size of the renderer
     *
     * @param width - Width in pixels
     * @param height - Height in pixels
     */
    setSize(width: number, height: number): void;
    /**
     * Render a video frame with the current shader
     *
     * @param frame - Video frame to render
     * @returns ImageData of the processed frame
     */
    render(frame: HTMLVideoElement | HTMLCanvasElement | ImageBitmap | ImageData): ImageData;
    /**
     * Set a custom fragment shader
     *
     * @param fragmentShaderSource - GLSL source code for the fragment shader
     */
    setFragmentShader(fragmentShaderSource: string): void;
    /**
     * Get the canvas element
     *
     * @returns Canvas element
     */
    getCanvas(): HTMLCanvasElement;
    /**
     * Clean up WebGL resources
     */
    dispose(): void;
}
