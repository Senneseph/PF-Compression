/**
 * Effect Chain System
 *
 * Allows multiple effects, encoders, decoders, and filters to be chained together
 * with intermediate frame capture and statistics tracking.
 */
import { Effect } from './effect';
import { Encoder } from '../webcam/encoders';
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
export declare class EffectChain {
    private stages;
    private captureIntermediates;
    /**
     * Create a new effect chain
     *
     * @param captureIntermediates - Whether to capture intermediate frames
     */
    constructor(captureIntermediates?: boolean);
    /**
     * Add an effect to the chain
     */
    addEffect(name: string, effect: Effect): void;
    /**
     * Add an encoder to the chain
     */
    addEncoder(name: string, encoder: Encoder): void;
    /**
     * Add a decoder to the chain
     */
    addDecoder(name: string, decoder: Decoder): void;
    /**
     * Add a filter to the chain
     */
    addFilter(name: string, filter: Filter): void;
    /**
     * Remove a stage by index
     */
    removeStage(index: number): void;
    /**
     * Enable or disable a stage
     */
    setStageEnabled(index: number, enabled: boolean): void;
    /**
     * Get all stages
     */
    getStages(): EffectChainStage[];
    /**
     * Clear all stages
     */
    clear(): void;
    /**
     * Calculate statistics for a frame
     */
    private calculateStatistics;
    /**
     * Process a frame through the effect chain
     */
    process(inputFrame: ImageData): ChainProcessingResult;
    /**
     * Clone an ImageData object
     */
    private cloneImageData;
    /**
     * Set whether to capture intermediate frames
     */
    setCaptureIntermediates(capture: boolean): void;
    /**
     * Get whether intermediate frames are being captured
     */
    getCaptureIntermediates(): boolean;
}
