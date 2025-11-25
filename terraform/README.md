# Terraform Deployment for PF-Compression PWA

This directory contains Terraform configuration to deploy the PF-Compression PWA to DigitalOcean.

## Prerequisites

1. **DigitalOcean Account**: Sign up at https://www.digitalocean.com
2. **DigitalOcean API Token**: Create at https://cloud.digitalocean.com/account/api/tokens
3. **Terraform**: Install from https://www.terraform.io/downloads
4. **SSH Key**: Generate if you don't have one: `ssh-keygen -t rsa -b 4096`
5. **Domain**: Ensure `iffuso.com` is added to your DigitalOcean account

## Quick Start

### 1. Configure Variables

Copy the example variables file and edit it:

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` and add your DigitalOcean API token:

```hcl
do_token = "your-actual-token-here"
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Review the Plan

```bash
terraform plan
```

This will show you what resources will be created.

### 4. Apply the Configuration

```bash
terraform apply
```

Type `yes` when prompted to confirm.

### 5. Get the Droplet IP

After deployment completes, Terraform will output the droplet's IP address:

```bash
terraform output droplet_ip
```

## What Gets Created

- **Droplet**: Ubuntu 20.04 with Docker pre-installed
- **Firewall**: Allows SSH (22), HTTP (80), and HTTPS (443)
- **DNS Record**: A record for `video-compression.iffuso.com`
- **SSH Key**: Your public key for secure access

## Post-Deployment Steps

### 1. SSH into the Droplet

```bash
ssh root@$(terraform output -raw droplet_ip)
```

### 2. Deploy the Application

On the droplet, clone your repository and deploy:

```bash
cd /opt/pf-compression
git clone https://github.com/Senneseph/PF-Compression.git .
cd docker
docker-compose -f docker-compose.pwa.yml up -d --build
```

### 3. Set Up SSL Certificate

```bash
# Install certbot if not already installed
apt-get update
apt-get install -y certbot python3-certbot-nginx

# Get SSL certificate
certbot --nginx -d video-compression.iffuso.com
```

### 4. Verify Deployment

Visit https://video-compression.iffuso.com in your browser.

## Manual Deployment (Alternative)

If you prefer to deploy manually without Terraform:

### 1. Create a Droplet

- Go to https://cloud.digitalocean.com/droplets/new
- Choose Ubuntu 20.04 with Docker
- Select size: Basic ($6/month)
- Add your SSH key
- Create droplet

### 2. Configure DNS

- Go to https://cloud.digitalocean.com/networking/domains
- Select `iffuso.com`
- Add A record:
  - Hostname: `video-compression`
  - Will Direct To: Your droplet's IP
  - TTL: 300 seconds

### 3. Deploy Application

SSH into your droplet and run:

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Docker and Docker Compose (if not pre-installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Clone repository
mkdir -p /opt/pf-compression
cd /opt/pf-compression
git clone https://github.com/Senneseph/PF-Compression.git .

# Build and deploy
cd docker
docker-compose -f docker-compose.pwa.yml up -d --build
```

### 4. Set Up SSL

```bash
apt-get install -y certbot python3-certbot-nginx
certbot --nginx -d video-compression.iffuso.com
```

## Updating the Application

To update the deployed application:

```bash
ssh root@video-compression.iffuso.com
cd /opt/pf-compression
git pull
cd docker
docker-compose -f docker-compose.pwa.yml up -d --build
```

## Destroying Resources

To remove all created resources:

```bash
terraform destroy
```

Type `yes` when prompted.

## Troubleshooting

### DNS Not Resolving

Wait a few minutes for DNS propagation. Check with:

```bash
dig video-compression.iffuso.com
```

### Container Not Starting

Check logs:

```bash
docker-compose -f docker-compose.pwa.yml logs
```

### SSL Certificate Issues

Ensure DNS is properly configured before running certbot.

## Cost Estimate

- Droplet (s-1vcpu-1gb): $6/month
- Bandwidth: Included (1TB)
- Total: ~$6/month

## Security Notes

- Keep your `terraform.tfvars` file secure (it's in .gitignore)
- Regularly update the droplet: `apt-get update && apt-get upgrade`
- Monitor firewall rules
- Use strong SSH keys
- Enable automatic security updates

## Support

For issues, check:
- DigitalOcean Status: https://status.digitalocean.com
- Terraform Docs: https://registry.terraform.io/providers/digitalocean/digitalocean/latest/docs
- GitHub Issues: https://github.com/Senneseph/PF-Compression/issues

