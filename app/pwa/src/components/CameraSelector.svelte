<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  
  const dispatch = createEventDispatcher();
  
  export let disabled = false;
  
  let cameras: MediaDeviceInfo[] = [];
  let selectedCamera: string = '';
  let loading = true;
  
  async function loadCameras() {
    try {
      loading = true;
      const devices = await navigator.mediaDevices.enumerateDevices();
      cameras = devices.filter(device => device.kind === 'videoinput');
      
      if (cameras.length > 0 && !selectedCamera) {
        selectedCamera = cameras[0].deviceId;
      }
    } catch (err) {
      console.error('Error loading cameras:', err);
    } finally {
      loading = false;
    }
  }
  
  function handleChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    selectedCamera = target.value;
    dispatch('change', selectedCamera);
  }
  
  onMount(() => {
    loadCameras();
  });
</script>

<div class="camera-selector">
  {#if loading}
    <div class="loading">Loading cameras...</div>
  {:else if cameras.length === 0}
    <div class="no-cameras">No cameras found</div>
  {:else}
    <select bind:value={selectedCamera} on:change={handleChange} {disabled}>
      {#each cameras as camera}
        <option value={camera.deviceId}>
          {camera.label || `Camera ${cameras.indexOf(camera) + 1}`}
        </option>
      {/each}
    </select>
    
    <button on:click={loadCameras} class="refresh-btn" {disabled}>
      🔄 Refresh
    </button>
  {/if}
</div>

<style>
  .camera-selector {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  select {
    width: 100%;
    padding: 0.75rem;
    font-size: 1rem;
    background: rgba(255, 255, 255, 0.1);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  select:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.15);
    border-color: rgba(255, 255, 255, 0.3);
  }
  
  select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  select option {
    background: #2d2d2d;
    color: white;
  }
  
  .refresh-btn {
    padding: 0.5rem 1rem;
    background: rgba(255, 255, 255, 0.1);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: all 0.2s;
  }
  
  .refresh-btn:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.3);
  }
  
  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .loading,
  .no-cameras {
    padding: 1rem;
    text-align: center;
    color: #aaa;
    font-style: italic;
  }
</style>

