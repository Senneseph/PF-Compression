<script lang="ts">
  export let fps: number = 0;
  export let effect: string = 'None';
  
  $: fpsColor = fps >= 25 ? '#4ade80' : fps >= 15 ? '#fbbf24' : '#ef4444';
</script>

<div class="stats">
  <div class="stat-item">
    <div class="stat-label">FPS</div>
    <div class="stat-value" style="color: {fpsColor}">
      {fps.toFixed(1)}
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
      640 × 480
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

