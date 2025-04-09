#!/bin/sh
# Script to initialize the PF-Compression project

# Check if Docker is installed
if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker is not installed. Please install Docker first."
  exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose >/dev/null 2>&1; then
  echo "Error: Docker Compose is not installed. Please install Docker Compose first."
  exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  if [ -f .env.default ]; then
    echo "Creating .env file from .env.default..."
    cp .env.default .env
  else
    echo "Error: .env.default file not found. Please create a .env file manually."
    exit 1
  fi
fi

# Make scripts executable
chmod +x docker/*.sh
chmod +x init.sh

# Build Docker images
echo "Building Docker images..."
docker-compose build

echo "Initialization completed successfully."
echo "You can now start the application with: docker-compose up -d"
echo "Or use the Makefile commands: make start"
exit 0
