#!/bin/bash

# Setup script for DigitalOcean droplet
# This script installs Docker, Docker Compose, and sets up the server

set -e

echo "🚀 Setting up PF-Compression server..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install required packages
echo "📦 Installing required packages..."
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git

# Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
else
    echo "Docker already installed"
fi

# Install Docker Compose
echo "🐳 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    echo "Docker Compose already installed"
fi

# Configure firewall
echo "🔥 Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# Create application directory
echo "📁 Creating application directory..."
sudo mkdir -p /opt/pf-compression
sudo chown $USER:$USER /opt/pf-compression

# Install Nginx for reverse proxy (if needed for SSL later)
echo "🌐 Installing Nginx..."
sudo apt-get install -y nginx

# Create deployment user
echo "👤 Setting up deployment..."
if ! id "deploy" &>/dev/null; then
    sudo useradd -m -s /bin/bash deploy
    sudo usermod -aG docker deploy
fi

echo "✅ Server setup complete!"
echo ""
echo "Next steps:"
if [ -n "$TARGET_DOMAIN" ]; then
    echo "1. Configure DNS to point ${TARGET_DOMAIN} to this server"
else
    echo "1. Configure DNS to point your domain to this server"
fi
echo "2. Run the deploy script to deploy the application"
echo "3. (Optional) Set up SSL with Let's Encrypt"

