<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { EffectChainStage } from 'pf-compression-pwa';
  
  const dispatch = createEventDispatcher();
  
  export let stages: EffectChainStage[] = [];
  
  // Available processors
  const availableEffects = [
    { name: 'Prime RGB', type: 'effect' },
    { name: 'Fibonacci RGB', type: 'effect' },
    { name: 'Color Negative', type: 'effect' },
    { name: 'Middle 4-Bit', type: 'effect' }
  ];
  
  const availableEncoders = [
    { name: 'RGB Strobe', type: 'encoder' },
    { name: 'RGB Even-Odd Strobe', type: 'encoder' },
    { name: 'RGB Matrix Strobe', type: 'encoder' }
  ];
  
  const availableDecoders = [
    { name: 'RGB Strobe', type: 'decoder' },
    { name: 'RGB Even-Odd Strobe', type: 'decoder' },
    { name: 'RGB Matrix Strobe', type: 'decoder' }
  ];
  
  const availableFilters = [
    { name: 'Color Negative', type: 'filter' },
    { name: 'Middle Four Bits', type: 'filter' },
    { name: 'Grayscale', type: 'filter' },
    { name: 'Threshold', type: 'filter' },
    { name: 'Brightness', type: 'filter' }
  ];
  
  let selectedType: 'effect' | 'encoder' | 'decoder' | 'filter' = 'effect';
  let selectedName = '';
  
  function addStage() {
    if (!selectedName) return;
    
    dispatch('addStage', {
      type: selectedType,
      name: selectedName
    });
    
    selectedName = '';
  }
  
  function removeStage(index: number) {
    dispatch('removeStage', index);
  }
  
  function toggleStage(index: number) {
    dispatch('toggleStage', index);
  }
  
  function moveUp(index: number) {
    if (index > 0) {
      dispatch('moveStage', { from: index, to: index - 1 });
    }
  }
  
  function moveDown(index: number) {
    if (index < stages.length - 1) {
      dispatch('moveStage', { from: index, to: index + 1 });
    }
  }
  
  $: availableOptions = selectedType === 'effect' ? availableEffects
    : selectedType === 'encoder' ? availableEncoders
    : selectedType === 'decoder' ? availableDecoders
    : availableFilters;
</script>

<div class="chain-builder">
  <h3>Effect Chain Builder</h3>
  
  <div class="add-stage">
    <select bind:value={selectedType}>
      <option value="effect">Effect</option>
      <option value="encoder">Encoder</option>
      <option value="decoder">Decoder</option>
      <option value="filter">Filter</option>
    </select>
    
    <select bind:value={selectedName}>
      <option value="">Select {selectedType}...</option>
      {#each availableOptions as option}
        <option value={option.name}>{option.name}</option>
      {/each}
    </select>
    
    <button on:click={addStage} disabled={!selectedName}>
      ➕ Add
    </button>
  </div>
  
  <div class="stages-list">
    {#if stages.length === 0}
      <div class="empty-state">
        No stages in chain. Add effects, encoders, decoders, or filters above.
      </div>
    {:else}
      {#each stages as stage, index}
        <div class="stage-item" class:disabled={!stage.enabled}>
          <div class="stage-info">
            <span class="stage-number">{index + 1}</span>
            <span class="stage-type" class:effect={stage.type === 'effect'}
                  class:encoder={stage.type === 'encoder'}
                  class:decoder={stage.type === 'decoder'}
                  class:filter={stage.type === 'filter'}>
              {stage.type}
            </span>
            <span class="stage-name">{stage.name}</span>
          </div>
          
          <div class="stage-controls">
            <button on:click={() => moveUp(index)} disabled={index === 0} title="Move up">
              ⬆
            </button>
            <button on:click={() => moveDown(index)} disabled={index === stages.length - 1} title="Move down">
              ⬇
            </button>
            <button on:click={() => toggleStage(index)} title={stage.enabled ? 'Disable' : 'Enable'}>
              {stage.enabled ? '👁' : '👁‍🗨'}
            </button>
            <button on:click={() => removeStage(index)} title="Remove">
              🗑
            </button>
          </div>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .chain-builder {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  h3 {
    margin: 0;
    color: #fff;
    font-size: 1.1rem;
  }

  .add-stage {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .add-stage select {
    flex: 1;
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    color: #fff;
    font-size: 0.9rem;
  }

  .add-stage button {
    padding: 0.5rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 4px;
    color: white;
    cursor: pointer;
    font-weight: 600;
    transition: transform 0.2s;
  }

  .add-stage button:hover:not(:disabled) {
    transform: scale(1.05);
  }

  .add-stage button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .stages-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-height: 400px;
    overflow-y: auto;
  }

  .empty-state {
    padding: 2rem;
    text-align: center;
    color: #888;
    font-style: italic;
  }

  .stage-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    transition: all 0.2s;
  }

  .stage-item:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.2);
  }

  .stage-item.disabled {
    opacity: 0.5;
  }

  .stage-info {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    flex: 1;
  }

  .stage-number {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 600;
    color: #aaa;
  }

  .stage-type {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .stage-type.effect {
    background: rgba(102, 126, 234, 0.3);
    color: #667eea;
  }

  .stage-type.encoder {
    background: rgba(74, 222, 128, 0.3);
    color: #4ade80;
  }

  .stage-type.decoder {
    background: rgba(251, 191, 36, 0.3);
    color: #fbbf24;
  }

  .stage-type.filter {
    background: rgba(239, 68, 68, 0.3);
    color: #ef4444;
  }

  .stage-name {
    color: #fff;
    font-weight: 500;
  }

  .stage-controls {
    display: flex;
    gap: 0.25rem;
  }

  .stage-controls button {
    padding: 0.25rem 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    color: #fff;
    cursor: pointer;
    font-size: 0.9rem;
    transition: all 0.2s;
  }

  .stage-controls button:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.2);
  }

  .stage-controls button:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }
</style>

