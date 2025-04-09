#!/bin/bash

# Script to build and run the PF-Compression PWA Docker container

# Function to display usage information
function show_usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -d, --dev     Run in development mode with hot reloading"
  echo "  -p, --prod    Run in production mode (default)"
  echo "  -b, --build   Build the Docker image"
  echo "  -r, --run     Run the Docker container"
  echo "  -h, --help    Show this help message"
  exit 1
}

# Default values
MODE="prod"
ACTION="both"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dev)
      MODE="dev"
      shift
      ;;
    -p|--prod)
      MODE="prod"
      shift
      ;;
    -b|--build)
      ACTION="build"
      shift
      ;;
    -r|--run)
      ACTION="run"
      shift
      ;;
    -h|--help)
      show_usage
      ;;
    *)
      echo "Unknown option: $1"
      show_usage
      ;;
  esac
done

# Set the Docker Compose file based on the mode
if [ "$MODE" == "dev" ]; then
  COMPOSE_FILE="docker-compose.dev.yml"
  echo "Running in development mode"
else
  COMPOSE_FILE="docker-compose.yml"
  echo "Running in production mode"
fi

# Execute the requested action
if [ "$ACTION" == "build" ] || [ "$ACTION" == "both" ]; then
  echo "Building Docker image..."
  docker-compose -f $COMPOSE_FILE build
fi

if [ "$ACTION" == "run" ] || [ "$ACTION" == "both" ]; then
  echo "Running Docker container..."
  docker-compose -f $COMPOSE_FILE up
fi
