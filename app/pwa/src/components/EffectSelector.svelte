<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  
  const dispatch = createEventDispatcher();
  
  export let currentEffect: string = 'None';
  
  const effects = [
    { value: 'None', label: 'None', description: 'No effect applied' },
    { value: 'Color Negative', label: 'Color Negative', description: 'Inverts all RGB values' },
    { value: 'Prime RGB', label: 'Prime RGB', description: 'Rounds RGB to nearest prime numbers' },
    { value: 'Fibonacci RGB', label: 'Fibonacci RGB', description: 'Rounds RGB to Fibonacci sequence' },
    { value: 'Middle 4-Bit', label: 'Middle 4-Bit', description: 'Preserves middle 4 bits only' }
  ];
  
  function handleChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    currentEffect = target.value;
    dispatch('change', currentEffect);
  }
</script>

<div class="effect-selector">
  <select bind:value={currentEffect} on:change={handleChange}>
    {#each effects as effect}
      <option value={effect.value}>{effect.label}</option>
    {/each}
  </select>
  
  <div class="effect-info">
    {#each effects as effect}
      {#if effect.value === currentEffect}
        <p class="description">{effect.description}</p>
      {/if}
    {/each}
  </div>
  
  <div class="effect-grid">
    {#each effects as effect}
      <button
        class="effect-card"
        class:active={effect.value === currentEffect}
        on:click={() => {
          currentEffect = effect.value;
          dispatch('change', currentEffect);
        }}
      >
        <div class="effect-name">{effect.label}</div>
        <div class="effect-desc">{effect.description}</div>
      </button>
    {/each}
  </div>
</div>

<style>
  .effect-selector {
    display: flex;
    flex-direction: column;
    gap: 1rem;
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
  
  select:hover {
    background: rgba(255, 255, 255, 0.15);
    border-color: rgba(255, 255, 255, 0.3);
  }
  
  select option {
    background: #2d2d2d;
    color: white;
  }
  
  .effect-info {
    min-height: 3rem;
  }
  
  .description {
    margin: 0;
    color: #aaa;
    font-size: 0.9rem;
    line-height: 1.5;
  }
  
  .effect-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.5rem;
  }
  
  .effect-card {
    padding: 0.75rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
  }
  
  .effect-card:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(102, 126, 234, 0.5);
    transform: translateX(4px);
  }
  
  .effect-card.active {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
    border-color: #667eea;
  }
  
  .effect-name {
    font-weight: 600;
    color: white;
    margin-bottom: 0.25rem;
  }
  
  .effect-desc {
    font-size: 0.8rem;
    color: #aaa;
  }
</style>

