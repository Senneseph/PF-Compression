# Quick Start Guide

## Getting Started in 3 Steps

### 1. Install Dependencies

Navigate to the PWA directory and install dependencies:

```bash
cd app/pwa
npm install
```

Or if you're using Bun:

```bash
cd app/pwa
bun install
```

### 2. Start Development Server

```bash
npm run dev
```

Or with Bun:

```bash
bun run dev
```

### 3. Open in Browser

Open your browser and navigate to:

```
http://localhost:2338
```

Click "Start Camera" and allow camera access when prompted.

## Using the Showcase

1. **Start Camera**: Click the play button to start your webcam
2. **Select Effect**: Choose from the available effects in the sidebar
3. **Switch Camera**: If you have multiple cameras, select from the dropdown
4. **Monitor Performance**: View FPS and other stats in real-time

## Available Effects

- **None**: Original video feed
- **Color Negative**: Inverts colors
- **Prime RGB**: Quantizes to prime numbers
- **Fibonacci RGB**: Quantizes to Fibonacci sequence
- **Middle 4-Bit**: Preserves middle 4 bits only

## Troubleshooting

### Camera Not Working

1. Ensure your browser has camera permissions
2. Check if another application is using the camera
3. Try refreshing the page
4. Check browser console for errors

### Build Errors

If you encounter build errors:

```bash
# Clean install
rm -rf node_modules
npm install

# Or with Bun
rm -rf node_modules
bun install
```

### Port Already in Use

If port 2338 is already in use, you can change it in `vite.config.ts`:

```typescript
server: {
  port: 3000, // Change to your preferred port
  host: true
}
```

## Building for Production

```bash
npm run build
```

The production build will be created in `../../dist/pwa`

## Docker Deployment

From the project root:

```bash
docker-compose up -d
```

## Next Steps

- Explore the source code in `src/`
- Check out the effects library in `../ts-pwalib/src/effects/`
- Read the main README.md for more details
- Contribute new effects!

## Support

For issues or questions, please visit:
https://github.com/Senneseph/PF-Compression

