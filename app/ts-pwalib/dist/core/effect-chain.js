/**
 * Effect Chain System
 *
 * Allows multiple effects, encoders, decoders, and filters to be chained together
 * with intermediate frame capture and statistics tracking.
 */
/**
 * Effect Chain class
 *
 * Manages a chain of effects, encoders, decoders, and filters
 */
export class EffectChain {
    /**
     * Create a new effect chain
     *
     * @param captureIntermediates - Whether to capture intermediate frames
     */
    constructor(captureIntermediates = true) {
        this.stages = [];
        this.captureIntermediates = true;
        this.captureIntermediates = captureIntermediates;
    }
    /**
     * Add an effect to the chain
     */
    addEffect(name, effect) {
        this.stages.push({
            name,
            type: 'effect',
            processor: effect,
            enabled: true
        });
    }
    /**
     * Add an encoder to the chain
     */
    addEncoder(name, encoder) {
        this.stages.push({
            name,
            type: 'encoder',
            processor: encoder,
            enabled: true
        });
    }
    /**
     * Add a decoder to the chain
     */
    addDecoder(name, decoder) {
        this.stages.push({
            name,
            type: 'decoder',
            processor: decoder,
            enabled: true
        });
    }
    /**
     * Add a filter to the chain
     */
    addFilter(name, filter) {
        this.stages.push({
            name,
            type: 'filter',
            processor: filter,
            enabled: true
        });
    }
    /**
     * Remove a stage by index
     */
    removeStage(index) {
        if (index >= 0 && index < this.stages.length) {
            this.stages.splice(index, 1);
        }
    }
    /**
     * Enable or disable a stage
     */
    setStageEnabled(index, enabled) {
        if (index >= 0 && index < this.stages.length) {
            this.stages[index].enabled = enabled;
        }
    }
    /**
     * Get all stages
     */
    getStages() {
        return [...this.stages];
    }
    /**
     * Clear all stages
     */
    clear() {
        this.stages = [];
    }
    /**
     * Calculate statistics for a frame
     */
    calculateStatistics(frame, stageName, stageType, processingTime) {
        const { width, height, data } = frame;
        const frameSize = data.length;
        // Calculate unique colors
        const colorSet = new Set();
        for (let i = 0; i < data.length; i += 4) {
            const color = `${data[i]},${data[i + 1]},${data[i + 2]}`;
            colorSet.add(color);
        }
        const uniqueColors = colorSet.size;
        // Calculate compression ratio (based on unique colors vs total pixels)
        const totalPixels = width * height;
        const compressionRatio = totalPixels / uniqueColors;
        // Calculate bit rate (bits per pixel)
        const bitsPerPixel = Math.log2(uniqueColors);
        const bitRate = bitsPerPixel * width * height;
        return {
            stageName,
            stageType: stageType,
            frameSize,
            uniqueColors,
            compressionRatio,
            bitRate,
            processingTime
        };
    }
    /**
     * Process a frame through the effect chain
     */
    process(inputFrame) {
        const startTime = performance.now();
        const intermediateFrames = [];
        const stageStatistics = [];
        let currentFrame = inputFrame;
        let encodedData = null;
        // Capture initial frame if requested
        if (this.captureIntermediates) {
            intermediateFrames.push(this.cloneImageData(currentFrame));
        }
        // Process through each stage
        for (const stage of this.stages) {
            if (!stage.enabled) {
                continue;
            }
            const stageStartTime = performance.now();
            try {
                switch (stage.type) {
                    case 'effect':
                        currentFrame = stage.processor.transform(currentFrame);
                        break;
                    case 'encoder':
                        encodedData = stage.processor.encode(currentFrame);
                        // For encoders, the output frame is in the encoded data
                        if (encodedData.outputFrame) {
                            currentFrame = encodedData.outputFrame;
                        }
                        break;
                    case 'decoder':
                        if (encodedData) {
                            currentFrame = stage.processor.decode(encodedData);
                        }
                        break;
                    case 'filter':
                        currentFrame = stage.processor.apply(currentFrame);
                        break;
                }
                const stageEndTime = performance.now();
                const processingTime = stageEndTime - stageStartTime;
                // Calculate statistics for this stage
                const stats = this.calculateStatistics(currentFrame, stage.name, stage.type, processingTime);
                stageStatistics.push(stats);
                // Capture intermediate frame if requested
                if (this.captureIntermediates) {
                    intermediateFrames.push(this.cloneImageData(currentFrame));
                }
            }
            catch (error) {
                console.error(`Error processing stage ${stage.name}:`, error);
            }
        }
        const endTime = performance.now();
        const totalProcessingTime = endTime - startTime;
        return {
            finalFrame: currentFrame,
            intermediateFrames,
            stageStatistics,
            totalProcessingTime
        };
    }
    /**
     * Clone an ImageData object
     */
    cloneImageData(imageData) {
        return new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
    }
    /**
     * Set whether to capture intermediate frames
     */
    setCaptureIntermediates(capture) {
        this.captureIntermediates = capture;
    }
    /**
     * Get whether intermediate frames are being captured
     */
    getCaptureIntermediates() {
        return this.captureIntermediates;
    }
}
