# Makefile for PF-Compression

# Default target
.PHONY: all
all: help

# Help target
.PHONY: help
help:
	@echo "PF-Compression Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make start        Start the application in production mode"
	@echo "  make start-dev    Start the application in development mode"
	@echo "  make stop         Stop the application"
	@echo "  make restart      Restart the application"
	@echo "  make logs         Show logs"
	@echo "  make build        Build the Docker images"
	@echo "  make test         Test if the application is working correctly"
	@echo "  make clean        Remove Docker containers and images"
	@echo "  make help         Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  SERVICE_PORT      Port where the service will be available (default: 2338)"
	@echo "  PROJECT_MODE      Project mode: 'serve' for normal operation, 'develop' for development (default: serve)"
	@echo ""

# Check environment
.PHONY: check-env
check-env:
	@sh docker/check-env.sh

# Start the application in production mode
.PHONY: start
start: check-env
	@echo "Starting PF-Compression in production mode..."
	@PROJECT_MODE=serve docker-compose up -d
	@echo "PF-Compression is running at http://localhost:$(shell grep SERVICE_PORT .env 2>/dev/null | cut -d= -f2 || echo 2338)"

# Start the application in development mode
.PHONY: start-dev
start-dev: check-env
	@echo "Starting PF-Compression in development mode..."
	@PROJECT_MODE=develop docker-compose -f docker-compose.yml up -d
	@echo "PF-Compression is running in development mode at http://localhost:$(shell grep SERVICE_PORT .env 2>/dev/null | cut -d= -f2 || echo 2338)"

# Stop the application
.PHONY: stop
stop:
	@echo "Stopping PF-Compression..."
	@docker-compose down

# Restart the application
.PHONY: restart
restart:
	@echo "Restarting PF-Compression..."
	@docker-compose restart

# Show logs
.PHONY: logs
logs:
	@docker-compose logs -f

# Build the Docker images
.PHONY: build
build:
	@echo "Building PF-Compression Docker images..."
	@docker-compose build

# Test if the application is working correctly
.PHONY: test
test:
	@echo "Testing if the application is working correctly..."
	@sh docker/test.sh

# Clean up Docker containers and images
.PHONY: clean
clean:
	@echo "Cleaning up PF-Compression Docker containers and images..."
	@docker-compose down --rmi all --volumes --remove-orphans
