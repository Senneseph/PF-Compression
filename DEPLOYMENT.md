# Deployment Guide for video-compression.iffuso.com

This guide will help you deploy the PF-Compression PWA to your DigitalOcean droplet.

## Option 1: Automated Deployment with Terraform (Recommended)

### Prerequisites

1. DigitalOcean API token from https://cloud.digitalocean.com/account/api/tokens
2. Terraform installed: https://www.terraform.io/downloads
3. SSH key generated: `ssh-keygen -t rsa -b 4096`
4. Domain `iffuso.com` added to your DigitalOcean account

### Steps

1. **Configure Terraform**

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` and add your DigitalOcean API token.

2. **Deploy Infrastructure**

```bash
terraform init
terraform plan
terraform apply
```

3. **Get Droplet IP**

```bash
terraform output droplet_ip
```

4. **Deploy Application**

SSH into the droplet:

```bash
ssh root@$(terraform output -raw droplet_ip)
```

On the droplet:

```bash
# Clone repository
cd /opt
git clone https://github.com/Senneseph/PF-Compression.git pf-compression
cd pf-compression

# Build and deploy
cd docker
docker-compose -f docker-compose.pwa.yml up -d --build
```

5. **Setup SSL**

```bash
certbot --nginx -d video-compression.iffuso.com
```

## Option 2: Manual Deployment

### 1. Create Droplet

- Go to DigitalOcean dashboard
- Create new droplet:
  - Image: Ubuntu 20.04 (Docker)
  - Size: Basic $6/month
  - Region: Choose closest to you
  - Add your SSH key
  - Create

### 2. Configure DNS

- Go to Networking > Domains > iffuso.com
- Add A record:
  - Hostname: `video-compression`
  - IP: Your droplet's IP
  - TTL: 300

### 3. Deploy Application

SSH into your droplet:

```bash
ssh root@YOUR_DROPLET_IP
```

Install dependencies:

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Docker (if not pre-installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

Clone and deploy:

```bash
# Clone repository
cd /opt
git clone https://github.com/Senneseph/PF-Compression.git pf-compression
cd pf-compression

# Build the application
cd app/ts-pwalib
npm install
npm run build
cd ../pwa
npm install
npm run build
cd ../..

# Deploy with Docker
cd docker
docker-compose -f docker-compose.pwa.yml up -d --build
```

### 4. Setup SSL Certificate

```bash
# Install Certbot
apt-get install -y certbot python3-certbot-nginx

# Get certificate
certbot --nginx -d video-compression.iffuso.com
```

## Option 3: Quick Deploy Script

Use the provided deployment script:

```bash
chmod +x deploy.sh
./deploy.sh
```

Follow the interactive prompts.

## Verification

After deployment, verify:

1. **HTTP Access**: http://video-compression.iffuso.com
2. **HTTPS Access**: https://video-compression.iffuso.com
3. **Health Check**: https://video-compression.iffuso.com/health

## Updating the Application

To update after making changes:

```bash
# On your local machine
git push

# On the droplet
ssh root@video-compression.iffuso.com
cd /opt/pf-compression
git pull
cd docker
docker-compose -f docker-compose.pwa.yml up -d --build
```

## Troubleshooting

### DNS Not Resolving

Wait 5-10 minutes for DNS propagation. Check with:

```bash
dig video-compression.iffuso.com
```

### Docker Container Not Starting

Check logs:

```bash
docker-compose -f docker-compose.pwa.yml logs
```

### SSL Certificate Issues

Ensure DNS is working before running certbot. Test with:

```bash
curl http://video-compression.iffuso.com/health
```

### Port Already in Use

Check what's using port 80:

```bash
netstat -tulpn | grep :80
```

## Monitoring

### View Logs

```bash
docker-compose -f docker-compose.pwa.yml logs -f
```

### Check Container Status

```bash
docker ps
```

### Resource Usage

```bash
docker stats
```

## Security Checklist

- [ ] SSH key authentication enabled
- [ ] Password authentication disabled
- [ ] Firewall configured (ports 22, 80, 443 only)
- [ ] SSL certificate installed
- [ ] Automatic security updates enabled
- [ ] Regular backups configured

## Cost

- Droplet: $6/month (1GB RAM, 1 vCPU)
- Bandwidth: Included (1TB)
- **Total: ~$6/month**

## Support

- GitHub Issues: https://github.com/Senneseph/PF-Compression/issues
- DigitalOcean Docs: https://docs.digitalocean.com
- Terraform Docs: https://registry.terraform.io/providers/digitalocean/digitalocean/latest/docs

