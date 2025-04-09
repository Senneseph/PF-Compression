# PF-Compression Docker Setup

This directory contains Docker configuration files for building and running the PF-Compression PWA.

## Files

- `Dockerfile`: Production Dockerfile for building and serving the PWA
- `Dockerfile.dev`: Development Dockerfile with hot reloading
- `nginx.conf`: Nginx configuration for serving the PWA
- `docker-compose.yml`: Docker Compose configuration for production
- `docker-compose.dev.yml`: Docker Compose configuration for development

## Usage

### Development

To start the development environment:

```bash
cd docker
docker-compose -f docker-compose.dev.yml up --build
```

This will:
1. Build the Docker image using `Dockerfile.dev`
2. Start a container with the development server
3. Mount the source code as a volume for hot reloading
4. Expose the application on port 2338

### Production

To build and run the production environment:

```bash
cd docker
docker-compose up --build
```

This will:
1. Build the Docker image using `Dockerfile`
2. Build the application
3. Serve the built application using Nginx
4. Expose the application on port 2338

## Configuration

### Port

The application is configured to run on port 2338. You can change this by modifying:
- The `EXPOSE` directive in the Dockerfiles
- The `listen` directive in `nginx.conf`
- The `ports` mapping in the docker-compose files

### Nginx

The Nginx configuration in `nginx.conf` includes:
- Gzip compression for better performance
- Cache control headers for static assets
- Security headers
- SPA routing support

## Volumes

In development mode, the source code is mounted as a volume to enable hot reloading. Changes to the source code will be immediately reflected in the running application.

## Environment Variables

- `NODE_ENV`: Set to `development` or `production` to control the build process
