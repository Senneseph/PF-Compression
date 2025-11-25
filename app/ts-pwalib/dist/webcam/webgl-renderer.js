/**
 * WebGL Renderer for GPU-accelerated video processing
 *
 * This module provides a WebGL-based renderer for applying effects to video frames.
 * It handles shader compilation, texture management, and rendering.
 */
/**
 * WebGL Renderer class for GPU-accelerated video processing
 */
export class WebGLRenderer {
    /**
     * Create a new WebGL Renderer
     *
     * @param canvas - Canvas element to render to, or create a new one if not provided
     */
    constructor(canvas) {
        this.gl = null;
        this.program = null;
        this.positionBuffer = null;
        this.texCoordBuffer = null;
        this.texture = null;
        this.framebuffer = null;
        this.outputTexture = null;
        this.width = 0;
        this.height = 0;
        this.canvas = canvas || document.createElement('canvas');
        this.initWebGL();
    }
    /**
     * Initialize WebGL context and resources
     */
    initWebGL() {
        // Try to get WebGL context
        this.gl = this.canvas.getContext('webgl') || this.canvas.getContext('experimental-webgl');
        if (!this.gl) {
            throw new Error('WebGL not supported');
        }
        // Create shader program
        const vertexShader = this.createShader(this.gl.VERTEX_SHADER, `
      attribute vec2 a_position;
      attribute vec2 a_texCoord;
      varying vec2 v_texCoord;
      
      void main() {
        gl_Position = vec4(a_position, 0, 1);
        v_texCoord = a_texCoord;
      }
    `);
        const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, `
      precision mediump float;
      uniform sampler2D u_image;
      varying vec2 v_texCoord;
      
      void main() {
        gl_FragColor = texture2D(u_image, v_texCoord);
      }
    `);
        this.program = this.createProgram(vertexShader, fragmentShader);
        // Create buffers
        this.positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([
            -1.0, -1.0,
            1.0, -1.0,
            -1.0, 1.0,
            1.0, 1.0,
        ]), this.gl.STATIC_DRAW);
        this.texCoordBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ]), this.gl.STATIC_DRAW);
        // Create texture
        this.texture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
        // Create framebuffer for offscreen rendering
        this.framebuffer = this.gl.createFramebuffer();
        this.outputTexture = this.gl.createTexture();
    }
    /**
     * Create a WebGL shader
     *
     * @param type - Shader type (VERTEX_SHADER or FRAGMENT_SHADER)
     * @param source - GLSL source code
     * @returns Compiled shader
     */
    createShader(type, source) {
        if (!this.gl) {
            throw new Error('WebGL not initialized');
        }
        const shader = this.gl.createShader(type);
        if (!shader) {
            throw new Error('Failed to create shader');
        }
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        const success = this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS);
        if (!success) {
            const info = this.gl.getShaderInfoLog(shader);
            this.gl.deleteShader(shader);
            throw new Error('Failed to compile shader: ' + info);
        }
        return shader;
    }
    /**
     * Create a WebGL program from shaders
     *
     * @param vertexShader - Vertex shader
     * @param fragmentShader - Fragment shader
     * @returns Linked program
     */
    createProgram(vertexShader, fragmentShader) {
        if (!this.gl) {
            throw new Error('WebGL not initialized');
        }
        const program = this.gl.createProgram();
        if (!program) {
            throw new Error('Failed to create program');
        }
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);
        const success = this.gl.getProgramParameter(program, this.gl.LINK_STATUS);
        if (!success) {
            const info = this.gl.getProgramInfoLog(program);
            this.gl.deleteProgram(program);
            throw new Error('Failed to link program: ' + info);
        }
        return program;
    }
    /**
     * Set the size of the renderer
     *
     * @param width - Width in pixels
     * @param height - Height in pixels
     */
    setSize(width, height) {
        this.width = width;
        this.height = height;
        this.canvas.width = width;
        this.canvas.height = height;
        if (!this.gl || !this.outputTexture) {
            return;
        }
        // Resize output texture
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.outputTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, width, height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
        // Set viewport
        this.gl.viewport(0, 0, width, height);
    }
    /**
     * Render a video frame with the current shader
     *
     * @param frame - Video frame to render
     * @returns ImageData of the processed frame
     */
    render(frame) {
        if (!this.gl || !this.program || !this.texture || !this.framebuffer || !this.outputTexture) {
            throw new Error('WebGL resources not initialized');
        }
        // Set size if needed
        let width = 0;
        let height = 0;
        if (frame instanceof HTMLVideoElement) {
            width = frame.videoWidth;
            height = frame.videoHeight;
        }
        else if (frame instanceof HTMLCanvasElement) {
            width = frame.width;
            height = frame.height;
        }
        else if (frame instanceof ImageBitmap) {
            width = frame.width;
            height = frame.height;
        }
        else if (frame instanceof ImageData) {
            width = frame.width;
            height = frame.height;
        }
        if (width !== this.width || height !== this.height) {
            this.setSize(width, height);
        }
        // Upload texture
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
        if (frame instanceof ImageData) {
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, width, height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, new Uint8Array(frame.data.buffer));
        }
        else {
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, frame);
        }
        // Set up framebuffer for offscreen rendering
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffer);
        this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.outputTexture, 0);
        // Use shader program
        this.gl.useProgram(this.program);
        // Set up vertex attributes
        const positionLocation = this.gl.getAttribLocation(this.program, 'a_position');
        const texCoordLocation = this.gl.getAttribLocation(this.program, 'a_texCoord');
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.enableVertexAttribArray(positionLocation);
        this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 0, 0);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
        this.gl.enableVertexAttribArray(texCoordLocation);
        this.gl.vertexAttribPointer(texCoordLocation, 2, this.gl.FLOAT, false, 0, 0);
        // Set uniforms
        const imageLocation = this.gl.getUniformLocation(this.program, 'u_image');
        this.gl.uniform1i(imageLocation, 0);
        // Draw
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        // Read pixels
        const pixels = new Uint8Array(this.width * this.height * 4);
        this.gl.readPixels(0, 0, this.width, this.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, pixels);
        // Reset framebuffer
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        // Create ImageData
        return new ImageData(new Uint8ClampedArray(pixels.buffer), this.width, this.height);
    }
    /**
     * Set a custom fragment shader
     *
     * @param fragmentShaderSource - GLSL source code for the fragment shader
     */
    setFragmentShader(fragmentShaderSource) {
        if (!this.gl) {
            throw new Error('WebGL not initialized');
        }
        // Create new shaders
        const vertexShader = this.createShader(this.gl.VERTEX_SHADER, `
      attribute vec2 a_position;
      attribute vec2 a_texCoord;
      varying vec2 v_texCoord;
      
      void main() {
        gl_Position = vec4(a_position, 0, 1);
        v_texCoord = a_texCoord;
      }
    `);
        const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, fragmentShaderSource);
        // Create new program
        const newProgram = this.createProgram(vertexShader, fragmentShader);
        // Delete old program
        if (this.program) {
            this.gl.deleteProgram(this.program);
        }
        // Use new program
        this.program = newProgram;
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
     * Clean up WebGL resources
     */
    dispose() {
        if (!this.gl) {
            return;
        }
        if (this.program) {
            this.gl.deleteProgram(this.program);
            this.program = null;
        }
        if (this.positionBuffer) {
            this.gl.deleteBuffer(this.positionBuffer);
            this.positionBuffer = null;
        }
        if (this.texCoordBuffer) {
            this.gl.deleteBuffer(this.texCoordBuffer);
            this.texCoordBuffer = null;
        }
        if (this.texture) {
            this.gl.deleteTexture(this.texture);
            this.texture = null;
        }
        if (this.outputTexture) {
            this.gl.deleteTexture(this.outputTexture);
            this.outputTexture = null;
        }
        if (this.framebuffer) {
            this.gl.deleteFramebuffer(this.framebuffer);
            this.framebuffer = null;
        }
        this.gl = null;
    }
}
