# Multi-stage build for PF-Compression PWA

# Stage 1: Build the TypeScript library
FROM node:20-alpine AS lib-builder

WORKDIR /app/ts-pwalib

# Copy library package files
COPY app/ts-pwalib/package*.json ./
RUN npm ci

# Copy library source
COPY app/ts-pwalib/ ./

# Build the library
RUN npm run build

# Stage 2: Build the PWA
FROM node:20-alpine AS pwa-builder

WORKDIR /app/pwa

# Copy PWA package files
COPY app/pwa/package*.json ./
RUN npm ci

# Copy built library from previous stage
COPY --from=lib-builder /app/ts-pwalib /app/ts-pwalib

# Copy PWA source
COPY app/pwa/ ./

# Build the PWA
RUN npm run build

# Stage 3: Production server with Nginx
FROM nginx:alpine

# Copy built PWA to nginx (build output is at /dist/pwa)
COPY --from=pwa-builder /dist/pwa /usr/share/nginx/html

# Copy nginx configuration
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]

