#!/bin/sh
# Script to check if the environment is properly set up

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

# Check if .env file exists, if not, create it from .env.default
if [ ! -f .env ]; then
  if [ -f .env.default ]; then
    echo "Creating .env file from .env.default..."
    cp .env.default .env
  else
    echo "Error: .env.default file not found. Please create a .env file manually."
    exit 1
  fi
fi

# Check if required environment variables are set
if ! grep -q "SERVICE_PORT" .env; then
  echo "Warning: SERVICE_PORT is not set in .env file. Using default value: 2338"
  echo "SERVICE_PORT=2338" >> .env
fi

if ! grep -q "PROJECT_MODE" .env; then
  echo "Warning: PROJECT_MODE is not set in .env file. Using default value: serve"
  echo "PROJECT_MODE=serve" >> .env
fi

# Check if ports are available
SERVICE_PORT=$(grep SERVICE_PORT .env | cut -d= -f2)
if lsof -Pi :$SERVICE_PORT -sTCP:LISTEN -t >/dev/null ; then
  echo "Warning: Port $SERVICE_PORT is already in use. You may need to change SERVICE_PORT in .env file."
fi

echo "Environment check completed successfully."
exit 0
