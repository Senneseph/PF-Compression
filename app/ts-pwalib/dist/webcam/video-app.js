/**
 * VideoApp class for the PF-Compression PWA
 *
 * This is the TypeScript equivalent of the Python VideoApp class in webcam.py.
 * It handles video capture, processing, and display.
 */
import { WebGLRenderer } from './webgl-renderer';
/**
 * VideoApp class for video capture, processing, and display
 */
export class VideoApp {
    /**
     * Create a new VideoApp
     *
     * @param options - Options for the VideoApp
     */
    constructor(options = {}) {
        this.isRunning = false;
        this.animationFrameId = 0;
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.fps = 0;
        this.stream = null;
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
        // Create WebGL renderer
        this.renderer = new WebGLRenderer(this.canvas);
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
    async start(deviceId) {
        if (this.isRunning) {
            return;
        }
        try {
            // Get user media
            const constraints = {
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
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play().then(() => resolve());
                };
            });
            // Start rendering
            this.isRunning = true;
            this.lastFrameTime = performance.now();
            this.frameCount = 0;
            this.renderFrame();
        }
        catch (error) {
            console.error('Error starting video:', error);
            throw error;
        }
    }
    /**
     * Stop the video app
     */
    stop() {
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
    renderFrame() {
        if (!this.isRunning) {
            return;
        }
        // Calculate FPS
        const now = performance.now();
        const elapsed = now - this.lastFrameTime;
        this.frameCount++;
        if (elapsed >= 1000) {
            this.fps = (this.frameCount * 1000) / elapsed;
            this.lastFrameTime = now;
            this.frameCount = 0;
        }
        // Process frame
        try {
            // Render the frame with WebGL
            const processedFrame = this.renderer.render(this.video);
            // Apply effect if one is set
            if (this.currentEffect) {
                const effectResult = this.currentEffect.transform(processedFrame);
                // Draw the effect result to the canvas
                const ctx = this.canvas.getContext('2d');
                if (ctx) {
                    ctx.putImageData(effectResult, 0, 0);
                }
            }
            // Draw FPS counter
            this.drawFPS();
        }
        catch (error) {
            console.error('Error rendering frame:', error);
        }
        // Request next frame
        this.animationFrameId = requestAnimationFrame(() => this.renderFrame());
    }
    /**
     * Draw FPS counter on the canvas
     */
    drawFPS() {
        const ctx = this.canvas.getContext('2d');
        if (!ctx) {
            return;
        }
        ctx.font = '16px Arial';
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        const text = `FPS: ${this.fps.toFixed(1)}`;
        ctx.strokeText(text, 10, 20);
        ctx.fillText(text, 10, 20);
    }
    /**
     * Set the current effect
     *
     * @param effect - Effect to apply, or null for no effect
     */
    setEffect(effect) {
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
    getEffect() {
        return this.currentEffect;
    }
    /**
     * Get the current FPS
     *
     * @returns Current frames per second
     */
    getFPS() {
        return this.fps;
    }
    /**
     * Get the canvas element
     *
     * @returns Canvas element
     */
    getCanvas() {
        return this.canvas;
    }
    /**
     * Get the video element
     *
     * @returns Video element
     */
    getVideo() {
        return this.video;
    }
    /**
     * Get available video devices
     *
     * @returns Promise that resolves to an array of video devices
     */
    static async getVideoDevices() {
        try {
            // Request permission to access devices
            await navigator.mediaDevices.getUserMedia({ video: true });
            // Get devices
            const devices = await navigator.mediaDevices.enumerateDevices();
            // Filter for video input devices
            return devices.filter(device => device.kind === 'videoinput');
        }
        catch (error) {
            console.error('Error getting video devices:', error);
            return [];
        }
    }
    /**
     * Take a snapshot of the current frame
     *
     * @returns Data URL of the snapshot
     */
    takeSnapshot() {
        return this.canvas.toDataURL('image/png');
    }
    /**
     * Clean up resources
     */
    dispose() {
        this.stop();
        this.renderer.dispose();
    }
}
