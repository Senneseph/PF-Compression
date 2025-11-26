/**
 * VideoApp class for the PF-Compression PWA
 * 
 * This is the TypeScript equivalent of the Python VideoApp class in webcam.py.
 * It handles video capture, processing, and display.
 */

import { WebGLRenderer } from './webgl-renderer';
import { Effect } from '../core/effect';
import { EffectChain, ChainProcessingResult, StageStatistics } from '../core/effect-chain';

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
export class VideoApp {
  private options: Required<VideoAppOptions>;
  private canvas: HTMLCanvasElement;
  private video: HTMLVideoElement;
  private renderer: WebGLRenderer;
  private currentEffect: Effect | null;
  private effectChain: EffectChain | null = null;
  private useEffectChain: boolean = false;
  private isRunning: boolean = false;
  private animationFrameId: number = 0;
  private lastFrameTime: number = 0;
  private frameCount: number = 0;
  private outputFps: number = 0;
  private inputLastFrameTime: number = 0;
  private inputFrameCount: number = 0;
  private inputFps: number = 0;
  private stream: MediaStream | null = null;
  private lastOriginalFrame: ImageData | null = null;
  private lastProcessedFrame: ImageData | null = null;
  private lastChainResult: ChainProcessingResult | null = null;

  /**
   * Create a new VideoApp
   *
   * @param options - Options for the VideoApp
   */
  constructor(options: VideoAppOptions = {}) {
    // Set default options
    this.options = {
      width: options.width || 640,
      height: options.height || 480,
      frameRate: options.frameRate || 30,
      initialEffect: options.initialEffect || null,
      canvas: options.canvas || document.createElement('canvas'),
      video: options.video || document.createElement('video')
    };

    // Set up canvas and video elements
    this.canvas = this.options.canvas;
    this.canvas.width = this.options.width;
    this.canvas.height = this.options.height;

    this.video = this.options.video;
    this.video.width = this.options.width;
    this.video.height = this.options.height;
    this.video.autoplay = true;
    this.video.muted = true;
    this.video.playsInline = true;

    // Create WebGL renderer with a separate canvas for processing
    this.renderer = new WebGLRenderer();
    this.renderer.setSize(this.options.width, this.options.height);

    // Set initial effect
    this.currentEffect = this.options.initialEffect;
  }

  /**
   * Start the video app
   * 
   * @param deviceId - Optional device ID for the camera
   * @returns Promise that resolves when the video is started
   */
  async start(deviceId?: string): Promise<void> {
    if (this.isRunning) {
      return;
    }

    try {
      // Get user media
      const constraints: MediaStreamConstraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId } }
          : {
              width: { ideal: this.options.width },
              height: { ideal: this.options.height },
              frameRate: { ideal: this.options.frameRate }
            },
        audio: false
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;

      // Wait for video to be ready
      await new Promise<void>((resolve) => {
        this.video.onloadedmetadata = () => {
          this.video.play().then(() => resolve());
        };
      });

      // Start rendering
      this.isRunning = true;
      this.lastFrameTime = performance.now();
      this.frameCount = 0;
      this.renderFrame();
    } catch (error) {
      console.error('Error starting video:', error);
      throw error;
    }
  }

  /**
   * Stop the video app
   */
  stop(): void {
    if (!this.isRunning) {
      return;
    }

    // Stop animation frame
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = 0;
    }

    // Stop video stream
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    this.video.srcObject = null;
    this.isRunning = false;
  }

  /**
   * Render a frame
   */
  private renderFrame(): void {
    if (!this.isRunning) {
      return;
    }

    // Calculate output FPS (processing rate)
    const now = performance.now();
    const elapsed = now - this.lastFrameTime;
    this.frameCount++;

    if (elapsed >= 1000) {
      this.outputFps = (this.frameCount * 1000) / elapsed;
      this.lastFrameTime = now;
      this.frameCount = 0;
    }

    // Calculate input FPS (camera frame rate)
    // Check if video has new frame data
    if (this.video.readyState >= this.video.HAVE_CURRENT_DATA) {
      const inputElapsed = now - this.inputLastFrameTime;
      this.inputFrameCount++;

      if (inputElapsed >= 1000) {
        this.inputFps = (this.inputFrameCount * 1000) / inputElapsed;
        this.inputLastFrameTime = now;
        this.inputFrameCount = 0;
      }
    }

    // Process frame
    try {
      // Render the frame with WebGL (this is the original frame)
      const originalFrame = this.renderer.render(this.video);
      this.lastOriginalFrame = originalFrame;

      let processedFrame: ImageData;

      // Use effect chain if enabled, otherwise use single effect
      if (this.useEffectChain && this.effectChain) {
        const chainResult = this.effectChain.process(originalFrame);
        this.lastChainResult = chainResult;
        processedFrame = chainResult.finalFrame;
      } else {
        // Apply effect if one is set, otherwise use the original frame
        processedFrame = this.currentEffect
          ? this.currentEffect.transform(originalFrame)
          : originalFrame;
      }

      this.lastProcessedFrame = processedFrame;

      // Draw the processed frame to the canvas
      const ctx = this.canvas.getContext('2d');
      if (ctx) {
        ctx.putImageData(processedFrame, 0, 0);
      }
    } catch (error) {
      console.error('Error rendering frame:', error);
    }

    // Request next frame
    this.animationFrameId = requestAnimationFrame(() => this.renderFrame());
  }

  /**
   * Set the current effect
   *
   * @param effect - Effect to apply, or null for no effect
   */
  setEffect(effect: Effect | null): void {
    this.currentEffect = effect;

    // Reset the effect if one is provided
    if (this.currentEffect) {
      this.currentEffect.reset();
    }
  }

  /**
   * Get the current effect
   *
   * @returns Current effect, or null if no effect is applied
   */
  getEffect(): Effect | null {
    return this.currentEffect;
  }

  /**
   * Get the current output FPS (processing rate)
   *
   * @returns Current output frames per second
   */
  getOutputFPS(): number {
    return this.outputFps;
  }

  /**
   * Get the current input FPS (camera frame rate)
   *
   * @returns Current input frames per second
   */
  getInputFPS(): number {
    return this.inputFps;
  }

  /**
   * Get the current FPS (output FPS for backward compatibility)
   *
   * @returns Current frames per second
   */
  getFPS(): number {
    return this.outputFps;
  }

  /**
   * Get the canvas element
   *
   * @returns Canvas element
   */
  getCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  /**
   * Get the video element
   * 
   * @returns Video element
   */
  getVideo(): HTMLVideoElement {
    return this.video;
  }

  /**
   * Get available video devices
   * 
   * @returns Promise that resolves to an array of video devices
   */
  static async getVideoDevices(): Promise<MediaDeviceInfo[]> {
    try {
      // Request permission to access devices
      await navigator.mediaDevices.getUserMedia({ video: true });
      
      // Get devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      
      // Filter for video input devices
      return devices.filter(device => device.kind === 'videoinput');
    } catch (error) {
      console.error('Error getting video devices:', error);
      return [];
    }
  }

  /**
   * Take a snapshot of the current frame
   *
   * @returns Data URL of the snapshot
   */
  takeSnapshot(): string {
    return this.canvas.toDataURL('image/png');
  }

  /**
   * Get the last original frame
   *
   * @returns Last original frame ImageData, or null if no frame has been captured
   */
  getOriginalFrame(): ImageData | null {
    return this.lastOriginalFrame;
  }

  /**
   * Get the last processed frame
   *
   * @returns Last processed frame ImageData, or null if no frame has been processed
   */
  getProcessedFrame(): ImageData | null {
    return this.lastProcessedFrame;
  }

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
  } {
    const width = this.options.width;
    const height = this.options.height;
    const originalSize = width * height * 4; // RGBA, 4 bytes per pixel

    // Estimate processed size based on effect
    let processedSize = originalSize;
    let compressionRatio = 1.0;

    if (this.currentEffect && this.lastProcessedFrame) {
      // Calculate actual data size (simplified estimation)
      // In a real scenario, you'd compress and measure the actual size
      const data = this.lastProcessedFrame.data;

      // Count unique colors to estimate compression
      const uniquePixels = new Set();
      for (let i = 0; i < data.length; i += 4) {
        const pixel = (data[i] << 24) | (data[i + 1] << 16) | (data[i + 2] << 8) | data[i + 3];
        uniquePixels.add(pixel);
      }

      // Estimate compression based on color reduction
      const colorReduction = uniquePixels.size / (width * height);
      processedSize = Math.floor(originalSize * colorReduction);
      compressionRatio = originalSize / processedSize;
    }

    return {
      fps: this.outputFps, // For backward compatibility
      inputFps: this.inputFps,
      outputFps: this.outputFps,
      width,
      height,
      originalSize,
      processedSize,
      compressionRatio,
      colorChannels: 4, // RGBA
      bitDepth: 8 // 8 bits per channel
    };
  }

  /**
   * Set the effect chain
   *
   * @param chain - Effect chain to use
   */
  setEffectChain(chain: EffectChain | null): void {
    this.effectChain = chain;
    this.useEffectChain = chain !== null;
  }

  /**
   * Get the effect chain
   *
   * @returns Current effect chain, or null if not using a chain
   */
  getEffectChain(): EffectChain | null {
    return this.effectChain;
  }

  /**
   * Get the last chain processing result
   *
   * @returns Last chain processing result, or null if not using a chain
   */
  getLastChainResult(): ChainProcessingResult | null {
    return this.lastChainResult;
  }

  /**
   * Get chain statistics
   *
   * @returns Array of stage statistics from the last chain processing
   */
  getChainStatistics(): StageStatistics[] {
    return this.lastChainResult?.stageStatistics || [];
  }

  /**
   * Get intermediate frames from the last chain processing
   *
   * @returns Array of intermediate frames
   */
  getIntermediateFrames(): ImageData[] {
    return this.lastChainResult?.intermediateFrames || [];
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.stop();
    this.renderer.dispose();
  }
}
