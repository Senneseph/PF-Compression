<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import { VideoApp } from 'pf-compression-pwa/src/webcam/video-app';
  import { 
    ColorNegativeEffect,
    PrimeRGBEffect,
    FibonacciRGBEffect,
    MiddleFourBitsEffect
  } from 'pf-compression-pwa/src/effects';
  
  const dispatch = createEventDispatcher();
  
  let canvas: HTMLCanvasElement;
  let video: HTMLVideoElement;
  let videoApp: VideoApp | null = null;
  let isRunning = false;
  let error: string | null = null;
  let fpsInterval: number;
  
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
      
      // Start FPS monitoring
      fpsInterval = window.setInterval(() => {
        if (videoApp) {
          dispatch('stats', { fps: videoApp.getFPS() });
        }
      }, 1000);
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
    
    if (wasRunning) {
      // Restart with new camera
      await start();
    }
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
  <div class="video-container">
    <canvas bind:this={canvas} width="640" height="480"></canvas>
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
  
  .video-container {
    position: relative;
    width: 100%;
    aspect-ratio: 4/3;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
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

