#!/bin/sh
# Script to test if the application is working correctly

# Get the service port from .env file or use default
SERVICE_PORT=$(grep SERVICE_PORT .env 2>/dev/null | cut -d= -f2 || echo 2338)

# Wait for the service to be available
echo "Waiting for the service to be available at http://localhost:$SERVICE_PORT..."
timeout=30
counter=0
while ! curl -s http://localhost:$SERVICE_PORT > /dev/null; do
  sleep 1
  counter=$((counter + 1))
  if [ $counter -ge $timeout ]; then
    echo "Error: Service did not become available within $timeout seconds."
    exit 1
  fi
done

# Test if the service is working correctly
echo "Testing if the service is working correctly..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$SERVICE_PORT)
if [ "$response" = "200" ]; then
  echo "Success: Service is working correctly."
  exit 0
else
  echo "Error: Service returned HTTP status code $response."
  exit 1
fi
