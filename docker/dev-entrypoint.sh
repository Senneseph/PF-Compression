#!/bin/sh
set -e

# Print configuration summary
echo "Starting PF-Compression PWA in development mode with the following configuration:"
echo "- PROJECT_MODE: ${PROJECT_MODE:-develop}"
echo "- SERVICE_PORT: ${SERVICE_PORT:-2338}"
echo "- ENABLE_HTTPS: ${ENABLE_HTTPS:-false}"
echo "- ENABLE_COMPRESSION: ${ENABLE_COMPRESSION:-false}"
echo "- LOG_LEVEL: ${LOG_LEVEL:-debug}"

# Install additional development dependencies if needed
if [ "$PROJECT_MODE" = "develop" ]; then
  echo "Checking for development dependencies..."
  if [ ! -d "node_modules/@types/node" ]; then
    echo "Installing development dependencies..."
    bun add -d typescript ts-node nodemon @types/node
  fi
fi

# Execute the command
exec "$@"
