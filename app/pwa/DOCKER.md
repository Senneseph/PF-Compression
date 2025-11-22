# Docker Deployment Guide

This guide explains how to deploy the PF-Compression Showcase PWA using Docker.

## Prerequisites

- Docker installed
- Docker Compose installed

## Quick Start

From the project root directory:

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

The application will be available at `http://localhost:2338`

## Development Mode

For development with hot reloading:

```bash
# From the docker directory
cd docker
docker-compose -f docker-compose.dev.yml up --build
```

This will:
- Mount source code as volumes
- Enable hot reloading
- Expose port 2338

## Production Mode

For production deployment:

```bash
# From the docker directory
cd docker
docker-compose up --build
```

This will:
- Build the application
- Serve with Nginx
- Optimize for production

## Environment Variables

You can configure the deployment using environment variables. Create a `.env` file in the project root:

```env
# Port configuration
SERVICE_PORT=2338

# Node environment
NODE_ENV=production

# Project mode (serve, develop, or pwa)
PROJECT_MODE=serve

# HTTPS configuration
ENABLE_HTTPS=false

# Compression
ENABLE_COMPRESSION=true

# Logging
LOG_LEVEL=info
```

## Custom Configuration

### Changing the Port

Edit `docker-compose.yml`:

```yaml
services:
  pf-compression-pwa:
    ports:
      - "3000:2338"  # Change 3000 to your desired port
```

### Volume Mounts

The following volumes are mounted:

```yaml
volumes:
  - ./app:/src/app              # Application source
  - ./build:/src/build          # Build output
  - ./dist:/src/dist            # Distribution files
```

## Building Custom Images

### Build the Development Image

```bash
docker build -f docker/Dockerfile.dev -t pf-compression-pwa:dev .
```

### Build the Production Image

```bash
docker build -f docker/Dockerfile -t pf-compression-pwa:latest .
```

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker-compose logs pf-compression-pwa
```

### Port Already in Use

Change the port in `docker-compose.yml` or stop the conflicting service.

### Permission Issues

On Linux, you may need to adjust file permissions:
```bash
sudo chown -R $USER:$USER app/ build/ dist/
```

### Rebuild After Changes

```bash
docker-compose down
docker-compose up --build
```

## Health Checks

The container includes health checks:

```bash
# Check container health
docker ps

# Manual health check
docker exec pf-compression-pwa curl -f http://localhost:2338 || exit 1
```

## Nginx Configuration

The production deployment uses Nginx. Configuration is in `docker/nginx.conf`.

Key features:
- Gzip compression
- Cache headers for static assets
- SPA routing support
- PWA manifest and service worker handling

## Security Considerations

For production deployment:

1. **HTTPS**: Enable HTTPS by setting `ENABLE_HTTPS=true`
2. **Firewall**: Configure firewall rules
3. **Updates**: Keep Docker images updated
4. **Secrets**: Don't commit `.env` files with sensitive data

## Monitoring

### View Logs

```bash
# All logs
docker-compose logs -f

# Specific service
docker-compose logs -f pf-compression-pwa
```

### Resource Usage

```bash
# Container stats
docker stats pf-compression-pwa
```

## Scaling

For multiple instances:

```bash
docker-compose up --scale pf-compression-pwa=3
```

Note: You'll need a load balancer for this to work properly.

## Cleanup

### Remove Containers

```bash
docker-compose down
```

### Remove Volumes

```bash
docker-compose down -v
```

### Remove Images

```bash
docker rmi pf-compression-pwa:latest
```

## Support

For issues, check:
- Docker logs: `docker-compose logs`
- Container status: `docker ps -a`
- GitHub issues: https://github.com/Senneseph/PF-Compression/issues

