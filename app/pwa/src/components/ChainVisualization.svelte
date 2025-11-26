<script lang="ts">
  import type { StageStatistics } from 'pf-compression-pwa';
  
  export let intermediateFrames: ImageData[] = [];
  export let stageStatistics: StageStatistics[] = [];
  export let stageNames: string[] = [];
  
  let canvases: HTMLCanvasElement[] = [];
  let selectedStage: number = -1;
  
  // Update canvases when frames change
  $: if (intermediateFrames.length > 0 && canvases.length > 0) {
    updateCanvases();
  }
  
  function updateCanvases() {
    intermediateFrames.forEach((frame, index) => {
      const canvas = canvases[index];
      if (canvas && frame) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          canvas.width = frame.width;
          canvas.height = frame.height;
          ctx.putImageData(frame, 0, 0);
        }
      }
    });
  }
  
  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
  
  function formatNumber(num: number): string {
    return num.toFixed(2);
  }
</script>

<div class="chain-visualization">
  <h3>Effect Chain Visualization</h3>
  
  {#if intermediateFrames.length === 0}
    <div class="empty-state">
      No chain processing results yet. Enable effect chain and start the camera.
    </div>
  {:else}
    <div class="frames-grid">
      {#each intermediateFrames as frame, index}
        <div class="frame-item" 
             class:selected={selectedStage === index}
             on:click={() => selectedStage = index}>
          <div class="frame-header">
            <span class="frame-number">{index === 0 ? 'Input' : `Stage ${index}`}</span>
            {#if index > 0 && stageNames[index - 1]}
              <span class="frame-name">{stageNames[index - 1]}</span>
            {/if}
          </div>
          <canvas bind:this={canvases[index]} class="frame-canvas"></canvas>
          {#if index > 0 && stageStatistics[index - 1]}
            <div class="frame-stats">
              <div class="stat-row">
                <span class="stat-label">Colors:</span>
                <span class="stat-value">{stageStatistics[index - 1].uniqueColors.toLocaleString()}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Ratio:</span>
                <span class="stat-value">{formatNumber(stageStatistics[index - 1].compressionRatio)}x</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Time:</span>
                <span class="stat-value">{formatNumber(stageStatistics[index - 1].processingTime)}ms</span>
              </div>
            </div>
          {/if}
        </div>
      {/each}
    </div>
    
    {#if selectedStage >= 0 && selectedStage < stageStatistics.length + 1}
      <div class="detailed-stats">
        <h4>
          {selectedStage === 0 ? 'Input Frame' : `Stage ${selectedStage}: ${stageNames[selectedStage - 1] || 'Unknown'}`}
        </h4>
        {#if selectedStage > 0 && stageStatistics[selectedStage - 1]}
          {@const stats = stageStatistics[selectedStage - 1]}
          <div class="stats-grid">
            <div class="stat-item">
              <div class="stat-label">Stage Type</div>
              <div class="stat-value type-badge" class:effect={stats.stageType === 'effect'}
                   class:encoder={stats.stageType === 'encoder'}
                   class:decoder={stats.stageType === 'decoder'}
                   class:filter={stats.stageType === 'filter'}>
                {stats.stageType}
              </div>
            </div>
            <div class="stat-item">
              <div class="stat-label">Frame Size</div>
              <div class="stat-value">{formatBytes(stats.frameSize)}</div>
            </div>
            <div class="stat-item">
              <div class="stat-label">Unique Colors</div>
              <div class="stat-value">{stats.uniqueColors.toLocaleString()}</div>
            </div>
            <div class="stat-item">
              <div class="stat-label">Compression Ratio</div>
              <div class="stat-value">{formatNumber(stats.compressionRatio)}x</div>
            </div>
            <div class="stat-item">
              <div class="stat-label">Bit Rate</div>
              <div class="stat-value">{formatBytes(stats.bitRate / 8)}</div>
            </div>
            <div class="stat-item">
              <div class="stat-label">Processing Time</div>
              <div class="stat-value">{formatNumber(stats.processingTime)} ms</div>
            </div>
          </div>
        {:else}
          <p>Original input frame from camera</p>
        {/if}
      </div>
    {/if}
  {/if}
</div>

<style>
  .chain-visualization {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  h3 {
    margin: 0;
    color: #fff;
    font-size: 1.1rem;
  }

  h4 {
    margin: 0 0 1rem 0;
    color: #fff;
    font-size: 1rem;
  }

  .empty-state {
    padding: 2rem;
    text-align: center;
    color: #888;
    font-style: italic;
  }

  .frames-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
    max-height: 500px;
    overflow-y: auto;
  }

  .frame-item {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.75rem;
    background: rgba(255, 255, 255, 0.05);
    border: 2px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .frame-item:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.2);
  }

  .frame-item.selected {
    border-color: #667eea;
    box-shadow: 0 0 12px rgba(102, 126, 234, 0.4);
  }

  .frame-header {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .frame-number {
    font-size: 0.75rem;
    font-weight: 600;
    color: #aaa;
    text-transform: uppercase;
  }

  .frame-name {
    font-size: 0.85rem;
    color: #fff;
    font-weight: 500;
  }

  .frame-canvas {
    width: 100%;
    height: auto;
    aspect-ratio: 4/3;
    background: #000;
    border-radius: 4px;
  }

  .frame-stats {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.75rem;
  }

  .stat-row {
    display: flex;
    justify-content: space-between;
  }

  .stat-row .stat-label {
    color: #888;
  }

  .stat-row .stat-value {
    color: #fff;
    font-weight: 600;
  }

  .detailed-stats {
    padding: 1rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
  }

  .stat-item {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .stat-item .stat-label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    font-weight: 600;
  }

  .stat-item .stat-value {
    font-size: 1rem;
    color: #fff;
    font-weight: 600;
  }

  .type-badge {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    text-transform: uppercase;
    width: fit-content;
  }

  .type-badge.effect {
    background: rgba(102, 126, 234, 0.3);
    color: #667eea;
  }

  .type-badge.encoder {
    background: rgba(74, 222, 128, 0.3);
    color: #4ade80;
  }

  .type-badge.decoder {
    background: rgba(251, 191, 36, 0.3);
    color: #fbbf24;
  }

  .type-badge.filter {
    background: rgba(239, 68, 68, 0.3);
    color: #ef4444;
  }
</style>

