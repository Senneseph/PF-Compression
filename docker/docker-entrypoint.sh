#!/bin/sh
set -e

# Function to substitute environment variables in a file
envsubst_file() {
  local file="$1"
  local tmpfile="$(mktemp)"
  
  # Create a list of environment variables to substitute
  local vars=$(env | cut -d= -f1 | grep -v '^_' | awk '{print "$"$0}' | tr '\n' ' ')
  
  # Substitute environment variables in the file
  envsubst "$vars" < "$file" > "$tmpfile"
  cat "$tmpfile" > "$file"
  rm "$tmpfile"
}

# Substitute environment variables in nginx configuration
echo "Configuring nginx for SERVICE_PORT=${SERVICE_PORT:-2338} and PROJECT_MODE=${PROJECT_MODE:-serve}"
envsubst_file /etc/nginx/conf.d/default.conf

# Print configuration summary
echo "Starting PF-Compression PWA with the following configuration:"
echo "- PROJECT_MODE: ${PROJECT_MODE:-serve}"
echo "- SERVICE_PORT: ${SERVICE_PORT:-2338}"
echo "- ENABLE_HTTPS: ${ENABLE_HTTPS:-false}"
echo "- ENABLE_COMPRESSION: ${ENABLE_COMPRESSION:-true}"
echo "- LOG_LEVEL: ${LOG_LEVEL:-info}"

# Execute the command
exec "$@"
