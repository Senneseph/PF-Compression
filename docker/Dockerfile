# Multi-stage build for PF-Compression PWA

# Stage 1: Build the application
FROM oven/bun:1.0 AS build

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY ts-pwalib/package*.json ./

# Install dependencies
RUN bun install

# Copy the source code
COPY ts-pwalib/ ./

# Build the application
RUN bun run build

# Stage 2: Serve the application with Nginx
FROM nginx:alpine

# Copy the built app from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom nginx configuration
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 2338
EXPOSE 2338

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
