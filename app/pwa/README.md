# PF-Compression Showcase PWA

A Progressive Web Application showcasing real-time webcam effects using the PF-Compression algorithms.

## Features

- 🎥 Real-time webcam capture and processing
- 🎨 Multiple video effects based on novel compression algorithms
- 📱 Progressive Web App (PWA) - installable and works offline
- ⚡ Built with Svelte + TypeScript for optimal performance
- 🎯 Responsive design for desktop and mobile

## Available Effects

- **None**: No effect applied (passthrough)
- **Color Negative**: Inverts all RGB values
- **Prime RGB**: Rounds RGB values to nearest prime numbers
- **Fibonacci RGB**: Rounds RGB values to Fibonacci sequence
- **Middle 4-Bit**: Preserves only the middle 4 bits of each pixel

## Development

### Prerequisites

- Node.js 18+ or Bun
- A webcam

### Installation

```bash
# Install dependencies
npm install
# or
bun install
```

### Running Locally

```bash
# Start development server
npm run dev
# or
bun run dev
```

The app will be available at `http://localhost:2338`

### Building for Production

```bash
# Build the app
npm run build
# or
bun run build
```

The built files will be in `../../dist/pwa`

### Preview Production Build

```bash
# Preview the production build
npm run preview
# or
bun run preview
```

## Docker Deployment

The PWA can be deployed using Docker:

```bash
# From the project root
docker-compose up -d
```

The app will be available at `http://localhost:2338`

## Technology Stack

- **Svelte 4**: Reactive UI framework
- **TypeScript**: Type-safe development
- **Vite**: Fast build tool and dev server
- **Vite PWA Plugin**: PWA capabilities
- **PF-Compression Library**: Custom video effects library

## Project Structure

```
app/pwa/
├── public/              # Static assets
├── src/
│   ├── components/      # Svelte components
│   │   ├── VideoPlayer.svelte
│   │   ├── EffectSelector.svelte
│   │   ├── CameraSelector.svelte
│   │   └── Stats.svelte
│   ├── App.svelte       # Main app component
│   ├── main.ts          # Entry point
│   └── app.css          # Global styles
├── index.html           # HTML template
├── vite.config.ts       # Vite configuration
├── tsconfig.json        # TypeScript configuration
└── package.json         # Dependencies
```

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

Note: Requires browser support for:
- WebRTC (getUserMedia)
- WebGL
- Service Workers (for PWA features)

## License

MIT

