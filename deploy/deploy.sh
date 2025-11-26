#!/bin/bash

# Deployment script for PF-Compression
# Deploys the application to the production server

set -e

# Configuration
SERVER_IP="${DEPLOY_SERVER_IP}"
if [ -z "$SERVER_IP" ]; then
    echo "ERROR: DEPLOY_SERVER_IP environment variable not set!"
    echo "Please set it in your .env file or environment"
    exit 1
fi

SERVER_USER="root"
APP_DIR="/opt/pf-compression"
REPO_URL="https://github.com/Senneseph/PF-Compression.git"

echo "🚀 Deploying PF-Compression to production..."

# Function to run commands on remote server
remote_exec() {
    ssh -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_IP} "$@"
}

# Check if we can connect to the server
echo "🔌 Testing connection to server..."
if ! remote_exec "echo 'Connection successful'"; then
    echo "❌ Failed to connect to server. Please check your SSH configuration."
    exit 1
fi

# Clone or update repository on server
echo "📥 Updating code on server..."
remote_exec "
    if [ -d ${APP_DIR}/.git ]; then
        cd ${APP_DIR}
        git fetch origin
        git reset --hard origin/main
    else
        sudo rm -rf ${APP_DIR}
        git clone ${REPO_URL} ${APP_DIR}
        cd ${APP_DIR}
    fi
"

# Build and deploy with Docker Compose
echo "🐳 Building and deploying containers..."
remote_exec "
    cd ${APP_DIR}
    docker-compose -f docker-compose.prod.yml down
    docker-compose -f docker-compose.prod.yml build --no-cache
    docker-compose -f docker-compose.prod.yml up -d
"

# Clean up old Docker images
echo "🧹 Cleaning up old Docker images..."
remote_exec "docker system prune -af --volumes"

# Check deployment status
echo "✅ Checking deployment status..."
sleep 5
if remote_exec "docker-compose -f ${APP_DIR}/docker-compose.prod.yml ps | grep -q 'Up'"; then
    echo "✅ Deployment successful!"
    echo ""
    echo "🌐 Application is now running at:"
    echo "   http://video-compression.iffuso.com"
    echo "   http://${SERVER_IP}"
else
    echo "❌ Deployment failed. Checking logs..."
    remote_exec "docker-compose -f ${APP_DIR}/docker-compose.prod.yml logs --tail=50"
    exit 1
fi

# Show logs
echo ""
echo "📋 Recent logs:"
remote_exec "docker-compose -f ${APP_DIR}/docker-compose.prod.yml logs --tail=20"

