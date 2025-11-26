# Deployment Guide for PF-Compression

## Prerequisites

1. **SSH Access**: You need SSH access to the DigitalOcean droplet
2. **Docker**: Will be installed automatically on the server
3. **DNS Configuration**: Point `$TARGET_DOMAIN` to `$DEPLOY_SERVER_IP`

## Quick Deployment

### Step 1: Set up SSH Access

You have two options:

#### Option A: Use DigitalOcean Console (Recommended)

1. Go to [DigitalOcean Console](https://cloud.digitalocean.com/droplets/530300735/console)
2. Login as `root` (use the password from DigitalOcean)
3. Run these commands:

```bash
# Generate SSH directory
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Add your public SSH key (get it from your local machine)
# On Windows PowerShell, run: Get-Content $env:USERPROFILE\.ssh\id_rsa.pub
# Then paste the output into this command:
echo "YOUR_SSH_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

#### Option B: Generate and Add SSH Key via PowerShell

If you don't have an SSH key:

```powershell
# Generate SSH key (if you don't have one)
ssh-keygen -t rsa -b 4096 -f "$env:USERPROFILE\.ssh\id_rsa"

# Display your public key
Get-Content "$env:USERPROFILE\.ssh\id_rsa.pub"
```

Then add it to the droplet using Option A above.

### Step 2: Deploy the Application

Once SSH access is configured, run:

```powershell
.\deploy\quick-deploy.ps1
```

This script will:
1. Install Docker and Docker Compose on the server
2. Configure the firewall
3. Build the application Docker image
4. Start the application

### Step 3: Configure DNS

Add an A record for `$TARGET_DOMAIN` pointing to `$DEPLOY_SERVER_IP`

## Manual Deployment Steps

If you prefer to deploy manually:

### 1. Setup Server

```bash
ssh root@$DEPLOY_SERVER_IP

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Configure firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Create app directory
mkdir -p /opt/pf-compression
```

### 2. Deploy Application

From your local machine:

```powershell
# Copy files to server
scp -r app Dockerfile docker-compose.prod.yml deploy root@$DEPLOY_SERVER_IP:/opt/pf-compression/

# SSH into server
ssh root@$DEPLOY_SERVER_IP

# Build and start
cd /opt/pf-compression
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

## Useful Commands

### View Logs
```bash
ssh root@$DEPLOY_SERVER_IP 'docker-compose -f /opt/pf-compression/docker-compose.prod.yml logs -f'
```

### Restart Application
```bash
ssh root@$DEPLOY_SERVER_IP 'docker-compose -f /opt/pf-compression/docker-compose.prod.yml restart'
```

### Stop Application
```bash
ssh root@$DEPLOY_SERVER_IP 'docker-compose -f /opt/pf-compression/docker-compose.prod.yml down'
```

### Update Application
```powershell
.\deploy\quick-deploy.ps1 -DeployOnly
```

## SSL/HTTPS Setup (Optional)

To enable HTTPS with Let's Encrypt:

```bash
ssh root@$DEPLOY_SERVER_IP

# Install Certbot
apt-get update
apt-get install -y certbot

# Stop the application temporarily
cd /opt/pf-compression
docker-compose -f docker-compose.prod.yml down

# Get SSL certificate
certbot certonly --standalone -d $TARGET_DOMAIN

# Update nginx.conf to use SSL
# Then restart the application
docker-compose -f docker-compose.prod.yml up -d
```

## Troubleshooting

### Cannot connect via SSH
- Check that your SSH key is added to the droplet
- Verify the droplet IP: `$DEPLOY_SERVER_IP`
- Try using DigitalOcean console access

### Application not starting
```bash
# Check logs
docker-compose -f /opt/pf-compression/docker-compose.prod.yml logs

# Check container status
docker-compose -f /opt/pf-compression/docker-compose.prod.yml ps

# Rebuild from scratch
docker-compose -f /opt/pf-compression/docker-compose.prod.yml down
docker system prune -af
docker-compose -f /opt/pf-compression/docker-compose.prod.yml build --no-cache
docker-compose -f /opt/pf-compression/docker-compose.prod.yml up -d
```

### Camera not working
- Ensure you're accessing via HTTPS (required for camera access in browsers)
- Check browser permissions for camera access

