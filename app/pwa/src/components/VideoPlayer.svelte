<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import {
    VideoApp,
    ColorNegativeEffect,
    PrimeRGBEffect,
    FibonacciRGBEffect,
    MiddleFourBitsEffect,
    EffectChain,
    type EffectChainStage,
    type ChainProcessingResult,
    // Encoders
    RGBStrobeEncoder,
    RGBEvenOddStrobeEncoder,
    RGBMatrixStrobeEncoder,
    // Decoders
    RGBStrobeDecoder,
    RGBEvenOddStrobeDecoder,
    RGBMatrixStrobeDecoder,
    // Filters
    ColorNegativeFilter,
    MiddleFourBitsFilter,
    GrayscaleFilter,
    ThresholdFilter,
    BrightnessFilter
  } from 'pf-compression-pwa';

  const dispatch = createEventDispatcher();

  let canvas: HTMLCanvasElement;
  let originalCanvas: HTMLCanvasElement;
  let video: HTMLVideoElement;
  let videoApp: VideoApp | null = null;
  let isRunning = false;
  let error: string | null = null;
  let fpsInterval: number;
  let statsInterval: number;
  let inputFps: number = 0;
  let outputFps: number = 0;
  
  const effects = {
    'None': null,
    'Color Negative': new ColorNegativeEffect(),
    'Prime RGB': new PrimeRGBEffect(),
    'Fibonacci RGB': new FibonacciRGBEffect(),
    'Middle 4-Bit': new MiddleFourBitsEffect()
  };
  
  export async function start() {
    if (isRunning) return;
    
    try {
      error = null;
      videoApp = new VideoApp({
        canvas,
        video,
        width: 640,
        height: 480,
        frameRate: 30
      });
      
      await videoApp.start();
      isRunning = true;
      dispatch('playing', true);

      // Start FPS and statistics monitoring
      fpsInterval = window.setInterval(() => {
        if (videoApp) {
          dispatch('stats', { fps: videoApp.getFPS() });
        }
      }, 1000);

      // Update original canvas with original frames and FPS
      statsInterval = window.setInterval(() => {
        if (videoApp && originalCanvas) {
          const originalFrame = videoApp.getOriginalFrame();
          if (originalFrame) {
            const ctx = originalCanvas.getContext('2d');
            if (ctx) {
              ctx.putImageData(originalFrame, 0, 0);
            }
          }

          // Update FPS values
          inputFps = videoApp.getInputFPS();
          outputFps = videoApp.getOutputFPS();

          // Dispatch detailed statistics
          const stats = videoApp.getStatistics();
          dispatch('detailedStats', stats);
        }
      }, 100); // Update at 10 FPS for smoother display
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to start camera';
      console.error('Error starting video:', err);
    }
  }
  
  export async function stop() {
    if (!isRunning) return;

    if (videoApp) {
      videoApp.stop();
      videoApp = null;
    }

    isRunning = false;
    dispatch('playing', false);

    if (fpsInterval) {
      clearInterval(fpsInterval);
    }

    if (statsInterval) {
      clearInterval(statsInterval);
    }
  }
  
  export function setEffect(effectName: string) {
    if (videoApp && effectName in effects) {
      videoApp.setEffect(effects[effectName as keyof typeof effects]);
    }
  }
  
  export async function setCamera(deviceId: string) {
    const wasRunning = isRunning;

    if (wasRunning) {
      await stop();
    }

    if (wasRunning && videoApp) {
      // Restart with new camera
      await videoApp.start(deviceId);
      isRunning = true;
      dispatch('playing', true);

      // Start FPS monitoring
      fpsInterval = window.setInterval(() => {
        if (videoApp) {
          dispatch('stats', { fps: videoApp.getFPS() });
        }
      }, 1000);
    }
  }

  // Effect chain methods
  let effectChain: EffectChain | null = null;

  export function setUseEffectChain(use: boolean) {
    if (use && !effectChain) {
      effectChain = new EffectChain();
    }
    if (videoApp) {
      videoApp.setEffectChain(use ? effectChain : null);
    }
  }

  export function addChainStage(type: 'effect' | 'encoder' | 'decoder' | 'filter', name: string) {
    if (!effectChain) {
      effectChain = new EffectChain();
    }

    switch (type) {
      case 'effect':
        if (name === 'Prime RGB') effectChain.addEffect(name, new PrimeRGBEffect());
        else if (name === 'Fibonacci RGB') effectChain.addEffect(name, new FibonacciRGBEffect());
        else if (name === 'Color Negative') effectChain.addEffect(name, new ColorNegativeEffect());
        else if (name === 'Middle 4-Bit') effectChain.addEffect(name, new MiddleFourBitsEffect());
        break;
      case 'encoder':
        if (name === 'RGB Strobe') effectChain.addEncoder(name, new RGBStrobeEncoder());
        else if (name === 'RGB Even-Odd Strobe') effectChain.addEncoder(name, new RGBEvenOddStrobeEncoder());
        else if (name === 'RGB Matrix Strobe') effectChain.addEncoder(name, new RGBMatrixStrobeEncoder());
        break;
      case 'decoder':
        if (name === 'RGB Strobe') effectChain.addDecoder(name, new RGBStrobeDecoder());
        else if (name === 'RGB Even-Odd Strobe') effectChain.addDecoder(name, new RGBEvenOddStrobeDecoder());
        else if (name === 'RGB Matrix Strobe') effectChain.addDecoder(name, new RGBMatrixStrobeDecoder());
        break;
      case 'filter':
        if (name === 'Color Negative') effectChain.addFilter(name, new ColorNegativeFilter());
        else if (name === 'Middle Four Bits') effectChain.addFilter(name, new MiddleFourBitsFilter());
        else if (name === 'Grayscale') effectChain.addFilter(name, new GrayscaleFilter());
        else if (name === 'Threshold') effectChain.addFilter(name, new ThresholdFilter(128));
        else if (name === 'Brightness') effectChain.addFilter(name, new BrightnessFilter(1.2));
        break;
    }

    if (videoApp) {
      videoApp.setEffectChain(effectChain);
    }
  }

  export function removeChainStage(index: number) {
    if (effectChain) {
      effectChain.removeStage(index);
      if (videoApp) {
        videoApp.setEffectChain(effectChain);
      }
    }
  }

  export function toggleChainStage(index: number) {
    if (effectChain) {
      const stages = effectChain.getStages();
      if (stages[index]) {
        effectChain.setStageEnabled(index, !stages[index].enabled);
        if (videoApp) {
          videoApp.setEffectChain(effectChain);
        }
      }
    }
  }

  export function moveChainStage(from: number, to: number) {
    if (effectChain) {
      const stages = effectChain.getStages();
      if (from >= 0 && from < stages.length && to >= 0 && to < stages.length) {
        const stage = stages[from];
        stages.splice(from, 1);
        stages.splice(to, 0, stage);
        // Rebuild chain with new order
        const newChain = new EffectChain();
        stages.forEach(s => {
          switch (s.type) {
            case 'effect':
              newChain.addEffect(s.name, s.processor as any);
              break;
            case 'encoder':
              newChain.addEncoder(s.name, s.processor as any);
              break;
            case 'decoder':
              newChain.addDecoder(s.name, s.processor as any);
              break;
            case 'filter':
              newChain.addFilter(s.name, s.processor as any);
              break;
          }
          if (!s.enabled) {
            newChain.setStageEnabled(newChain.getStages().length - 1, false);
          }
        });
        effectChain = newChain;
        if (videoApp) {
          videoApp.setEffectChain(effectChain);
        }
      }
    }
  }

  export function getChainStages(): EffectChainStage[] {
    return effectChain?.getStages() || [];
  }

  export function getLastChainResult(): ChainProcessingResult | null {
    return videoApp?.getLastChainResult() || null;
  }

  onMount(() => {
    // Auto-start on mount
    start();
  });
  
  onDestroy(() => {
    stop();
  });
</script>

<div class="video-player">
  <div class="video-grid">
    <div class="video-wrapper">
      <div class="video-header">
        <span class="video-title">Original</span>
        <span class="fps-badge" class:active={isRunning}>
          {inputFps.toFixed(1)} FPS
        </span>
      </div>
      <div class="video-container">
        <canvas bind:this={originalCanvas} width="640" height="480"></canvas>
      </div>
    </div>

    <div class="video-wrapper">
      <div class="video-header">
        <span class="video-title">Processed</span>
        <span class="fps-badge" class:active={isRunning}>
          {outputFps.toFixed(1)} FPS
        </span>
      </div>
      <div class="video-container">
        <canvas bind:this={canvas} width="640" height="480"></canvas>
      </div>
    </div>

    <video bind:this={video} style="display: none;"></video>

    {#if error}
      <div class="error-overlay">
        <p>⚠️ {error}</p>
        <button on:click={start}>Retry</button>
      </div>
    {/if}

    {#if !isRunning && !error}
      <div class="start-overlay">
        <button class="start-button" on:click={start}>
          <span class="icon">▶</span>
          Start Camera
        </button>
      </div>
    {/if}
  </div>

  <div class="controls">
    <button on:click={isRunning ? stop : start} class="control-btn">
      {isRunning ? '⏸ Pause' : '▶ Play'}
    </button>
  </div>
</div>

<style>
  .video-player {
    width: 100%;
  }

  .video-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    position: relative;
  }

  .video-wrapper {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .video-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.75rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .video-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #aaa;
  }

  .fps-badge {
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    color: #666;
    transition: all 0.3s ease;
  }

  .fps-badge.active {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
  }

  .video-container {
    position: relative;
    width: 100%;
    aspect-ratio: 4/3;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid #333;
    border-radius: 8px;
    overflow: hidden;
  }

  canvas {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  
  .error-overlay,
  .start-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.8);
    backdrop-filter: blur(10px);
    grid-column: 1 / -1;
    z-index: 20;
  }

  .error-overlay p {
    color: #ff6b6b;
    font-size: 1.2rem;
    margin-bottom: 1rem;
  }

  .start-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem 2rem;
    font-size: 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 50px;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
  }

  .start-button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
  }

  .start-button .icon {
    font-size: 1.2rem;
  }

  .controls {
    padding: 1rem;
    display: flex;
    justify-content: center;
    gap: 1rem;
    background: rgba(0, 0, 0, 0.5);
  }

  .control-btn {
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    background: rgba(255, 255, 255, 0.1);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .control-btn:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.3);
  }

  button {
    padding: 0.5rem 1rem;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 1rem;
  }

  button:hover {
    background: #5568d3;
  }
</style>

