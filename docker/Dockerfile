# Multi-stage build for PF-Compression PWA

# Accept build arguments
ARG PROJECT_MODE=serve

# Stage 1: Build the application
FROM oven/bun:1.0 AS build

# Set working directory
WORKDIR /src

# Set build arguments as environment variables
ARG PROJECT_MODE
ENV PROJECT_MODE=${PROJECT_MODE}

# Copy package.json and package-lock.json
COPY app/ts-pwalib/package*.json ./ts-pwalib/

# Install dependencies
WORKDIR /src/ts-pwalib
RUN bun install
WORKDIR /src

# Copy the source code
COPY app/ ./app/

# Build the application
WORKDIR /src/app/ts-pwalib
RUN echo "Building in ${PROJECT_MODE} mode" && \
    bun run build

# Copy built files to the build directory
WORKDIR /src
RUN mkdir -p /src/build
RUN cp -r /src/app/ts-pwalib/dist/* /src/build/

# Stage 2: Serve the application with Nginx
FROM cgr.dev/chainguard/nginx:latest

# Set environment variables
ARG PROJECT_MODE
ENV PROJECT_MODE=${PROJECT_MODE}

# Copy the built app from the build stage
COPY --from=build /src/build /usr/share/nginx/html

# Copy custom nginx configuration
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

# Create a script to substitute environment variables in nginx config
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Expose port 2338
EXPOSE 2338

# Start Nginx with our custom entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]



