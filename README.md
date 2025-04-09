# Phi-Fibonacci Compression

The goal of this project is to create a novel compression algorithm that works on all data types.

## Quick Start

The easiest way to get started is to use Docker:

```bash
# Clone the repository
git clone https://github.com/Senneseph/PF-Compression.git
cd PF-Compression

# Initialize the project (optional)
./init.sh

# Start the application
docker-compose up -d
```

The application will be available at http://localhost:2338

## Configuration

You can configure the application by creating a `.env` file in the root directory. The following environment variables are available:

```bash
# Port where the service will be available
SERVICE_PORT=2338

# Project mode: "serve" for normal operation, "develop" for development
PROJECT_MODE=serve
```

## Development

To start the application in development mode:

```bash
# Using docker-compose
PROJECT_MODE=develop docker-compose up -d

# Using make
make start-dev
```

## Available Commands

The project includes a Makefile with the following commands:

```bash
# Start the application in production mode
make start

# Start the application in development mode
make start-dev

# Stop the application
make stop

# Restart the application
make restart

# Show logs
make logs

# Build the Docker images
make build

# Test if the application is working correctly
make test

# Clean up Docker containers and images
make clean

# Show help
make help
```

## Project Structure

- `/lib`: Abstract library interfaces
- `/research-effects-library`: Concrete effect implementations
- `/examples`: Example scripts
- `/ts-pwalib`: TypeScript PWA library
- `/docker`: Docker configuration
- `/docs`: Documentation
