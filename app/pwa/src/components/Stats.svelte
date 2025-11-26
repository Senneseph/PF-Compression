<script lang="ts">
  export let fps: number = 0;
  export let inputFps: number = 0;
  export let outputFps: number = 0;
  export let effect: string = 'None';
  export let width: number = 640;
  export let height: number = 480;
  export let originalSize: number = 0;
  export let processedSize: number = 0;
  export let compressionRatio: number = 1.0;
  export let colorChannels: number = 4;
  export let bitDepth: number = 8;

  $: inputFpsColor = inputFps >= 25 ? '#4ade80' : inputFps >= 15 ? '#fbbf24' : '#ef4444';
  $: outputFpsColor = outputFps >= 25 ? '#4ade80' : outputFps >= 15 ? '#fbbf24' : '#ef4444';
  $: compressionColor = compressionRatio > 2 ? '#4ade80' : compressionRatio > 1.5 ? '#fbbf24' : '#ef4444';
  $: fpsAcceleration = inputFps > 0 ? (outputFps / inputFps) : 1.0;
  $: accelerationColor = fpsAcceleration >= 1.0 ? '#4ade80' : '#fbbf24';

  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
</script>

<div class="stats">
  <div class="stat-item">
    <div class="stat-label">Input FPS (Camera)</div>
    <div class="stat-value" style="color: {inputFpsColor}">
      {inputFps.toFixed(1)}
    </div>
  </div>

  <div class="stat-item">
    <div class="stat-label">Output FPS (Processed)</div>
    <div class="stat-value" style="color: {outputFpsColor}">
      {outputFps.toFixed(1)}
    </div>
  </div>

  <div class="stat-item">
    <div class="stat-label">FPS Acceleration</div>
    <div class="stat-value" style="color: {accelerationColor}">
      {fpsAcceleration.toFixed(2)}x
    </div>
  </div>

  <div class="stat-item">
    <div class="stat-label">Current Effect</div>
    <div class="stat-value effect-name">
      {effect}
    </div>
  </div>

  <div class="stat-item">
    <div class="stat-label">Resolution</div>
    <div class="stat-value">
      {width} × {height}
    </div>
  </div>

  <div class="stat-item">
    <div class="stat-label">Color Depth</div>
    <div class="stat-value">
      {colorChannels} channels × {bitDepth}-bit
    </div>
  </div>

  <div class="stat-item">
    <div class="stat-label">Original Frame Size</div>
    <div class="stat-value">
      {formatBytes(originalSize)}
    </div>
  </div>

  <div class="stat-item">
    <div class="stat-label">Processed Frame Size</div>
    <div class="stat-value">
      {formatBytes(processedSize)}
    </div>
  </div>

  <div class="stat-item">
    <div class="stat-label">Compression Ratio</div>
    <div class="stat-value" style="color: {compressionColor}">
      {compressionRatio.toFixed(2)}x
    </div>
  </div>

  <div class="stat-item">
    <div class="stat-label">Size Reduction</div>
    <div class="stat-value" style="color: {compressionColor}">
      {((1 - processedSize / originalSize) * 100).toFixed(1)}%
    </div>
  </div>
</div>

<style>
  .stats {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  
  .stat-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .stat-label {
    color: #aaa;
    font-size: 0.9rem;
    font-weight: 500;
  }
  
  .stat-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: white;
  }
  
  .effect-name {
    font-size: 0.95rem;
    text-align: right;
  }
</style>

