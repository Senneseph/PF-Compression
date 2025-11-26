/**
 * Effect Chain System
 * 
 * Allows multiple effects, encoders, decoders, and filters to be chained together
 * with intermediate frame capture and statistics tracking.
 */

import { Effect } from './effect';
import { Encoder, EncodedData } from '../webcam/encoders';
import { Decoder } from '../webcam/decoders';
import { Filter } from '../webcam/filters';

/**
 * Statistics for a single stage in the effect chain
 */
export interface StageStatistics {
  stageName: string;
  stageType: 'effect' | 'encoder' | 'decoder' | 'filter';
  frameSize: number;
  uniqueColors: number;
  compressionRatio: number;
  bitRate: number;
  processingTime: number;
}

/**
 * A single stage in the effect chain
 */
export interface EffectChainStage {
  name: string;
  type: 'effect' | 'encoder' | 'decoder' | 'filter';
  processor: Effect | Encoder | Decoder | Filter;
  enabled: boolean;
}

/**
 * Result of processing through the effect chain
 */
export interface ChainProcessingResult {
  finalFrame: ImageData;
  intermediateFrames: ImageData[];
  stageStatistics: StageStatistics[];
  totalProcessingTime: number;
}

/**
 * Effect Chain class
 * 
 * Manages a chain of effects, encoders, decoders, and filters
 */
export class EffectChain {
  private stages: EffectChainStage[] = [];
  private captureIntermediates: boolean = true;
  
  /**
   * Create a new effect chain
   * 
   * @param captureIntermediates - Whether to capture intermediate frames
   */
  constructor(captureIntermediates: boolean = true) {
    this.captureIntermediates = captureIntermediates;
  }
  
  /**
   * Add an effect to the chain
   */
  addEffect(name: string, effect: Effect): void {
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
  addEncoder(name: string, encoder: Encoder): void {
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
  addDecoder(name: string, decoder: Decoder): void {
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
  addFilter(name: string, filter: Filter): void {
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
  removeStage(index: number): void {
    if (index >= 0 && index < this.stages.length) {
      this.stages.splice(index, 1);
    }
  }
  
  /**
   * Enable or disable a stage
   */
  setStageEnabled(index: number, enabled: boolean): void {
    if (index >= 0 && index < this.stages.length) {
      this.stages[index].enabled = enabled;
    }
  }
  
  /**
   * Get all stages
   */
  getStages(): EffectChainStage[] {
    return [...this.stages];
  }
  
  /**
   * Clear all stages
   */
  clear(): void {
    this.stages = [];
  }
  
  /**
   * Calculate statistics for a frame
   */
  private calculateStatistics(
    frame: ImageData,
    stageName: string,
    stageType: string,
    processingTime: number
  ): StageStatistics {
    const { width, height, data } = frame;
    const frameSize = data.length;
    
    // Calculate unique colors
    const colorSet = new Set<string>();
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
      stageType: stageType as any,
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
  process(inputFrame: ImageData): ChainProcessingResult {
    const startTime = performance.now();
    const intermediateFrames: ImageData[] = [];
    const stageStatistics: StageStatistics[] = [];

    let currentFrame = inputFrame;
    let encodedData: EncodedData | null = null;

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
            currentFrame = (stage.processor as Effect).transform(currentFrame);
            break;

          case 'encoder':
            encodedData = (stage.processor as Encoder).encode(currentFrame);
            // For encoders, the output frame is in the encoded data
            if (encodedData.outputFrame) {
              currentFrame = encodedData.outputFrame;
            }
            break;

          case 'decoder':
            if (encodedData) {
              currentFrame = (stage.processor as Decoder).decode(encodedData);
            }
            break;

          case 'filter':
            currentFrame = (stage.processor as Filter).apply(currentFrame);
            break;
        }

        const stageEndTime = performance.now();
        const processingTime = stageEndTime - stageStartTime;

        // Calculate statistics for this stage
        const stats = this.calculateStatistics(
          currentFrame,
          stage.name,
          stage.type,
          processingTime
        );
        stageStatistics.push(stats);

        // Capture intermediate frame if requested
        if (this.captureIntermediates) {
          intermediateFrames.push(this.cloneImageData(currentFrame));
        }
      } catch (error) {
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
  private cloneImageData(imageData: ImageData): ImageData {
    return new ImageData(
      new Uint8ClampedArray(imageData.data),
      imageData.width,
      imageData.height
    );
  }

  /**
   * Set whether to capture intermediate frames
   */
  setCaptureIntermediates(capture: boolean): void {
    this.captureIntermediates = capture;
  }

  /**
   * Get whether intermediate frames are being captured
   */
  getCaptureIntermediates(): boolean {
    return this.captureIntermediates;
  }
}
