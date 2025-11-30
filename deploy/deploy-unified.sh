#!/bin/bash

# Consolidated Deployment Script for PF-Compression
# Builds Docker image locally and deploys to DigitalOcean droplet
# Optionally sets up nginx reverse proxy and SSL

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration from environment
SERVER_IP="${DEPLOY_SERVER_IP}"
TARGET_DOMAIN="${TARGET_DOMAIN}"
SERVER_USER="${SERVER_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/a-icon-deploy}"
IMAGE_NAME="pf-compression-web"
IMAGE_TAG="latest"
CONTAINER_NAME="pf-compression-web"
CONTAINER_PORT="8080"

# Flags
SETUP_NGINX=false
SETUP_SSL=false
SKIP_BUILD=false
SHOW_HELP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-nginx)
            SETUP_NGINX=true
            shift
            ;;
        --setup-ssl)
            SETUP_SSL=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            SHOW_HELP=true
            shift
            ;;
    esac
done

# Show help
if [ "$SHOW_HELP" = true ]; then
    echo -e "${CYAN}PF-Compression Deployment Script${NC}"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --setup-nginx    Setup nginx reverse proxy on the server"
    echo "  --setup-ssl      Setup SSL certificate with Let's Encrypt"
    echo "  --skip-build     Skip Docker build (use existing image)"
    echo "  --help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DEPLOY_SERVER_IP    - Server IP address (required)"
    echo "  TARGET_DOMAIN       - Domain name (required for nginx/ssl)"
    echo "  SERVER_USER         - SSH user (default: root)"
    echo "  SSH_KEY             - SSH key path (default: ~/.ssh/a-icon-deploy)"
    echo ""
    echo "Examples:"
    echo "  $0                              # Deploy only"
    echo "  $0 --setup-nginx                # Deploy and setup nginx"
    echo "  $0 --setup-nginx --setup-ssl    # Full setup with SSL"
    echo "  $0 --skip-build                 # Deploy without rebuilding"
    exit 0
fi

# Validate environment variables
if [ -z "$SERVER_IP" ]; then
    echo -e "${RED}ERROR: DEPLOY_SERVER_IP environment variable not set!${NC}"
    echo "Please set it in your .env file or environment"
    exit 1
fi

# Print header
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         PF-Compression Deployment Script                  ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Server IP:    $SERVER_IP"
[ -n "$TARGET_DOMAIN" ] && echo "Domain:       $TARGET_DOMAIN"
echo "Setup Nginx:  $SETUP_NGINX"
echo "Setup SSL:    $SETUP_SSL"
echo "Skip Build:   $SKIP_BUILD"
echo ""

# Test SSH connection
echo -e "${YELLOW}Testing SSH connection...${NC}"
if ! ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SERVER_USER}@${SERVER_IP}" "echo 'Connected'" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Cannot connect to server!${NC}"
    echo "Please check:"
    echo "  1. Server IP is correct: $SERVER_IP"
    echo "  2. SSH key exists at: $SSH_KEY"
    echo "  3. Server is accessible"
    exit 1
fi
echo -e "${GREEN}✓ SSH connection successful${NC}"
echo ""

# Step 1: Build Docker image locally
if [ "$SKIP_BUILD" = false ]; then
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN} Step 1/4: Building Docker Image${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    if ! docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .; then
        echo ""
        echo -e "${RED}ERROR: Docker build failed!${NC}"
        exit 1
    fi
    echo ""
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping Docker build (using existing image)...${NC}"
    echo ""
fi

# Step 2: Save image to tar file
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN} Step 2/4: Packaging Docker Image${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""

TEMP_FILE="/tmp/pf-compression-image.tar"
echo "Saving image to: $TEMP_FILE"
if ! docker save -o "$TEMP_FILE" "${IMAGE_NAME}:${IMAGE_TAG}"; then
    echo ""
    echo -e "${RED}ERROR: Failed to save Docker image!${NC}"
    exit 1
fi

FILE_SIZE=$(du -h "$TEMP_FILE" | cut -f1)
echo -e "${GREEN}✓ Image saved: $FILE_SIZE${NC}"
echo ""

# Step 3: Upload image to server
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN} Step 3/4: Uploading to Server${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo "Uploading image to ${SERVER_IP}..."
if ! scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o Compression=yes "$TEMP_FILE" "${SERVER_USER}@${SERVER_IP}:/tmp/pf-compression-image.tar"; then
    echo ""
    echo -e "${RED}ERROR: Failed to upload image!${NC}"
    rm -f "$TEMP_FILE"
    exit 1
fi

echo -e "${GREEN}✓ Upload complete${NC}"
echo ""

# Step 4: Deploy on server
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN} Step 4/4: Deploying Container${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Create deployment script
DEPLOY_SCRIPT=$(cat <<'DEPLOY_EOF'
#!/bin/bash
set -e

CONTAINER_NAME="$1"
IMAGE_NAME="$2"
IMAGE_TAG="$3"
CONTAINER_PORT="$4"

echo "Stopping existing container..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "Loading new image..."
docker load -i /tmp/pf-compression-image.tar

echo "Removing temporary file..."
rm /tmp/pf-compression-image.tar

echo "Starting container..."
docker run -d --name "$CONTAINER_NAME" --restart unless-stopped -p "${CONTAINER_PORT}:80" "${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "Container status:"
docker ps | grep "$CONTAINER_NAME"
DEPLOY_EOF
)

if ! ssh -i "$SSH_KEY" "${SERVER_USER}@${SERVER_IP}" bash -s "$CONTAINER_NAME" "$IMAGE_NAME" "$IMAGE_TAG" "$CONTAINER_PORT" <<< "$DEPLOY_SCRIPT"; then
    echo ""
    echo -e "${RED}ERROR: Failed to deploy container!${NC}"
    rm -f "$TEMP_FILE"
    exit 1
fi

# Cleanup local tar file
rm -f "$TEMP_FILE"

echo ""
echo -e "${GREEN}✓ Container deployed successfully${NC}"
echo ""

# Setup Nginx if requested
if [ "$SETUP_NGINX" = true ]; then
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN} Setting up Nginx Reverse Proxy${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    if [ -z "$TARGET_DOMAIN" ]; then
        echo -e "${RED}ERROR: TARGET_DOMAIN environment variable not set!${NC}"
        echo "Nginx setup requires a domain name."
        exit 1
    fi

    # Prepare nginx configuration
    echo "Preparing nginx configuration for $TARGET_DOMAIN..."
    TEMP_CONF="/tmp/nginx-${TARGET_DOMAIN}.conf"
    sed "s/video-compression\.iffuso\.com/${TARGET_DOMAIN}/g" deploy/nginx-host.conf > "$TEMP_CONF"

    # Upload configuration
    echo "Uploading nginx configuration..."
    TEMP_CONF_NAME=$(echo "$TARGET_DOMAIN" | tr '.' '-')
    if ! scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$TEMP_CONF" "${SERVER_USER}@${SERVER_IP}:/tmp/${TEMP_CONF_NAME}.conf"; then
        echo -e "${RED}ERROR: Failed to upload nginx config!${NC}"
        rm -f "$TEMP_CONF"
        exit 1
    fi

    rm -f "$TEMP_CONF"

    # Install and configure nginx
    echo "Configuring nginx on server..."

    NGINX_SCRIPT=$(cat <<'NGINX_EOF'
#!/bin/bash
set -e

DOMAIN="$1"
TEMP_CONF_NAME="$2"

# Install nginx if not already installed
if ! command -v nginx &> /dev/null; then
    apt-get update
    apt-get install -y nginx
fi

# Copy configuration
cp "/tmp/${TEMP_CONF_NAME}.conf" "/etc/nginx/sites-available/${DOMAIN}"

# Enable site
ln -sf "/etc/nginx/sites-available/${DOMAIN}" /etc/nginx/sites-enabled/

# Test nginx configuration
nginx -t

# Reload nginx
systemctl reload nginx

# Ensure nginx is enabled and running
systemctl enable nginx
systemctl start nginx

echo ""
echo "Nginx configured successfully!"
NGINX_EOF
)

    if ! ssh -i "$SSH_KEY" "${SERVER_USER}@${SERVER_IP}" bash -s "$TARGET_DOMAIN" "$TEMP_CONF_NAME" <<< "$NGINX_SCRIPT"; then
        echo -e "${RED}ERROR: Failed to configure nginx!${NC}"
        exit 1
    fi

    echo ""
    echo -e "${GREEN}✓ Nginx reverse proxy configured${NC}"
    echo ""
fi

# Setup SSL if requested
if [ "$SETUP_SSL" = true ]; then
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN} Setting up SSL Certificate${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    if [ -z "$TARGET_DOMAIN" ]; then
        echo -e "${RED}ERROR: TARGET_DOMAIN environment variable not set!${NC}"
        echo "SSL setup requires a domain name."
        exit 1
    fi

    # Verify DNS
    echo "Verifying DNS configuration for $TARGET_DOMAIN..."
    DNS_RESULT=$(nslookup "$TARGET_DOMAIN" 2>&1 | grep "Address:" | tail -1 | awk '{print $2}')
    echo "DNS resolves to: $DNS_RESULT"

    if [ "$DNS_RESULT" != "$SERVER_IP" ]; then
        echo ""
        echo -e "${YELLOW}WARNING: DNS is not pointing to $SERVER_IP${NC}"
        echo "SSL certificate may fail. Continue anyway? (y/N)"
        read -r CONTINUE
        if [ "$CONTINUE" != "y" ]; then
            echo "Skipping SSL setup."
            exit 0
        fi
    fi

    # Install certbot and obtain certificate
    echo "Installing certbot and obtaining SSL certificate..."

    SSL_SCRIPT=$(cat <<'SSL_EOF'
#!/bin/bash
set -e

DOMAIN="$1"

# Install certbot if not already installed
if ! command -v certbot &> /dev/null; then
    apt-get update
    apt-get install -y certbot python3-certbot-nginx
fi

# Obtain SSL certificate
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email admin@iffuso.com --redirect

echo ""
echo "SSL certificate obtained successfully!"
certbot certificates
SSL_EOF
)

    if ! ssh -i "$SSH_KEY" "${SERVER_USER}@${SERVER_IP}" bash -s "$TARGET_DOMAIN" <<< "$SSL_SCRIPT"; then
        echo ""
        echo -e "${RED}ERROR: Failed to obtain SSL certificate!${NC}"
        echo ""
        echo "Common issues:"
        echo "  1. DNS not propagated yet (wait 5-30 minutes)"
        echo "  2. DNS pointing to wrong IP address"
        echo "  3. Port 80 not accessible from internet"
        exit 1
    fi

    echo ""
    echo -e "${GREEN}✓ SSL certificate configured${NC}"
    echo ""
fi

# Final summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Deployment Complete!                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Application URLs:${NC}"
echo "  Direct IP:  http://${SERVER_IP}:${CONTAINER_PORT}"

if [ -n "$TARGET_DOMAIN" ]; then
    if [ "$SETUP_SSL" = true ]; then
        echo "  Domain:     https://${TARGET_DOMAIN}"
    else
        echo "  Domain:     http://${TARGET_DOMAIN}"
        if [ "$SETUP_NGINX" = true ]; then
            echo "              (Run with --setup-ssl to enable HTTPS)"
        fi
    fi
fi

echo ""
echo -e "${CYAN}Useful commands:${NC}"
echo "  View logs:     ssh -i $SSH_KEY ${SERVER_USER}@${SERVER_IP} 'docker logs -f $CONTAINER_NAME'"
echo "  Restart:       ssh -i $SSH_KEY ${SERVER_USER}@${SERVER_IP} 'docker restart $CONTAINER_NAME'"
echo "  Stop:          ssh -i $SSH_KEY ${SERVER_USER}@${SERVER_IP} 'docker stop $CONTAINER_NAME'"
echo ""

