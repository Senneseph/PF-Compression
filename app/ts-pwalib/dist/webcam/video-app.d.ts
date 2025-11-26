/**
 * VideoApp class for the PF-Compression PWA
 *
 * This is the TypeScript equivalent of the Python VideoApp class in webcam.py.
 * It handles video capture, processing, and display.
 */
import { Effect } from '../core/effect';
/**
 * Options for the VideoApp
 */
export interface VideoAppOptions {
    /** Width of the video */
    width?: number;
    /** Height of the video */
    height?: number;
    /** Target frame rate */
    frameRate?: number;
    /** Initial effect to apply */
    initialEffect?: Effect | null;
    /** Canvas element to render to */
    canvas?: HTMLCanvasElement;
    /** Video element to use */
    video?: HTMLVideoElement;
}
/**
 * VideoApp class for video capture, processing, and display
 */
export declare class VideoApp {
    private options;
    private canvas;
    private video;
    private renderer;
    private currentEffect;
    private isRunning;
    private animationFrameId;
    private lastFrameTime;
    private frameCount;
    private outputFps;
    private inputLastFrameTime;
    private inputFrameCount;
    private inputFps;
    private stream;
    private lastOriginalFrame;
    private lastProcessedFrame;
    /**
     * Create a new VideoApp
     *
     * @param options - Options for the VideoApp
     */
    constructor(options?: VideoAppOptions);
    /**
     * Start the video app
     *
     * @param deviceId - Optional device ID for the camera
     * @returns Promise that resolves when the video is started
     */
    start(deviceId?: string): Promise<void>;
    /**
     * Stop the video app
     */
    stop(): void;
    /**
     * Render a frame
     */
    private renderFrame;
    /**
     * Set the current effect
     *
     * @param effect - Effect to apply, or null for no effect
     */
    setEffect(effect: Effect | null): void;
    /**
     * Get the current effect
     *
     * @returns Current effect, or null if no effect is applied
     */
    getEffect(): Effect | null;
    /**
     * Get the current output FPS (processing rate)
     *
     * @returns Current output frames per second
     */
    getOutputFPS(): number;
    /**
     * Get the current input FPS (camera frame rate)
     *
     * @returns Current input frames per second
     */
    getInputFPS(): number;
    /**
     * Get the current FPS (output FPS for backward compatibility)
     *
     * @returns Current frames per second
     */
    getFPS(): number;
    /**
     * Get the canvas element
     *
     * @returns Canvas element
     */
    getCanvas(): HTMLCanvasElement;
    /**
     * Get the video element
     *
     * @returns Video element
     */
    getVideo(): HTMLVideoElement;
    /**
     * Get available video devices
     *
     * @returns Promise that resolves to an array of video devices
     */
    static getVideoDevices(): Promise<MediaDeviceInfo[]>;
    /**
     * Take a snapshot of the current frame
     *
     * @returns Data URL of the snapshot
     */
    takeSnapshot(): string;
    /**
     * Get the last original frame
     *
     * @returns Last original frame ImageData, or null if no frame has been captured
     */
    getOriginalFrame(): ImageData | null;
    /**
     * Get the last processed frame
     *
     * @returns Last processed frame ImageData, or null if no frame has been processed
     */
    getProcessedFrame(): ImageData | null;
    /**
     * Get statistics about the current frames
     *
     * @returns Object containing frame statistics
     */
    getStatistics(): {
        fps: number;
        inputFps: number;
        outputFps: number;
        width: number;
        height: number;
        originalSize: number;
        processedSize: number;
        compressionRatio: number;
        colorChannels: number;
        bitDepth: number;
    };
    /**
     * Clean up resources
     */
    dispose(): void;
}
