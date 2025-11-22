<script lang="ts">
  import VideoPlayer from './components/VideoPlayer.svelte';
  import EffectSelector from './components/EffectSelector.svelte';
  import CameraSelector from './components/CameraSelector.svelte';
  import Stats from './components/Stats.svelte';
  
  let videoPlayerRef: VideoPlayer;
  let currentEffect: string = 'None';
  let fps: number = 0;
  let isPlaying: boolean = false;
  
  function handleEffectChange(event: CustomEvent<string>) {
    currentEffect = event.detail;
    if (videoPlayerRef) {
      videoPlayerRef.setEffect(currentEffect);
    }
  }
  
  function handleCameraChange(event: CustomEvent<string>) {
    if (videoPlayerRef) {
      videoPlayerRef.setCamera(event.detail);
    }
  }
  
  function handleStatsUpdate(event: CustomEvent<{ fps: number }>) {
    fps = event.detail.fps;
  }
  
  function handlePlayingChange(event: CustomEvent<boolean>) {
    isPlaying = event.detail;
  }
</script>

<main>
  <header>
    <h1>🎥 PF-Compression Showcase</h1>
    <p>Real-time webcam effects using novel compression algorithms</p>
  </header>
  
  <div class="container">
    <div class="video-section">
      <VideoPlayer 
        bind:this={videoPlayerRef}
        on:stats={handleStatsUpdate}
        on:playing={handlePlayingChange}
      />
    </div>
    
    <div class="controls-section">
      <div class="control-group">
        <h3>Camera</h3>
        <CameraSelector on:change={handleCameraChange} disabled={isPlaying} />
      </div>
      
      <div class="control-group">
        <h3>Effect</h3>
        <EffectSelector on:change={handleEffectChange} currentEffect={currentEffect} />
      </div>
      
      <div class="control-group">
        <h3>Stats</h3>
        <Stats {fps} effect={currentEffect} />
      </div>
    </div>
  </div>
  
  <footer>
    <p>
      Built with Svelte + TypeScript | 
      <a href="https://github.com/Senneseph/PF-Compression" target="_blank" rel="noopener">
        View on GitHub
      </a>
    </p>
  </footer>
</main>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    color: #ffffff;
    min-height: 100vh;
  }
  
  main {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
  }
  
  header {
    text-align: center;
    margin-bottom: 2rem;
  }
  
  h1 {
    font-size: 2.5rem;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  header p {
    color: #aaa;
    font-size: 1.1rem;
  }
  
  .container {
    display: grid;
    grid-template-columns: 1fr 350px;
    gap: 2rem;
    margin-bottom: 2rem;
  }
  
  @media (max-width: 1024px) {
    .container {
      grid-template-columns: 1fr;
    }
  }
  
  .video-section {
    background: #000;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  }
  
  .controls-section {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }
  
  .control-group {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .control-group h3 {
    margin: 0 0 1rem 0;
    font-size: 1.2rem;
    color: #667eea;
  }
  
  footer {
    text-align: center;
    padding: 2rem 0;
    color: #888;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  footer a {
    color: #667eea;
    text-decoration: none;
  }
  
  footer a:hover {
    text-decoration: underline;
  }
</style>

