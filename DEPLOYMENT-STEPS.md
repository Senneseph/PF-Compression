# 🚀 Deployment Steps for video-compression.iffuso.com

## Current Status
- ✅ Droplet found: `a-icon-app` (ID: 530300735)
- ✅ IP Address: `$DEPLOY_SERVER_IP`
- ✅ Deployment files created
- ⏳ SSH access needed
- ⏳ DNS configuration needed

## Step 1: Get Your SSH Public Key

Open PowerShell and run:

```powershell
# If you don't have an SSH key, generate one:
ssh-keygen -t rsa -b 4096 -f "$env:USERPROFILE\.ssh\id_rsa"

# Display your public key (copy this entire output):
Get-Content "$env:USERPROFILE\.ssh\id_rsa.pub"
```

**Copy the entire output** - it should look like:
```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC... your-email@example.com
```

## Step 2: Add SSH Key to Droplet

I've opened the DigitalOcean console for you in your browser.

1. **Login to the console** (use your root password from DigitalOcean)
2. **Run these commands** in the console:

```bash
# Create SSH directory
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Add your SSH key (paste the key you copied from Step 1)
echo "PASTE_YOUR_SSH_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys

# Set correct permissions
chmod 600 ~/.ssh/authorized_keys

# Verify it was added
cat ~/.ssh/authorized_keys
```

## Step 3: Test SSH Connection

Back in PowerShell on your local machine:

```powershell
# Test SSH connection
ssh root@$DEPLOY_SERVER_IP "echo 'SSH works!'"
```

If you see "SSH works!", you're ready to deploy!

## Step 4: Deploy the Application

Run the deployment script:

```powershell
cd d:\Projects\PROMPTCRAFTIAN-IO\PF-Compression
.\deploy\quick-deploy.ps1
```

This will:
- ✅ Install Docker on the server
- ✅ Install Docker Compose
- ✅ Configure firewall (ports 22, 80, 443)
- ✅ Build the application
- ✅ Start the application

## Step 5: Configure DNS

Add an **A record** for your domain:

- **Host**: `video-compression` (or `@` if using subdomain)
- **Type**: `A`
- **Value**: `$DEPLOY_SERVER_IP`
- **TTL**: `3600` (or automatic)

Where to configure:
- If using DigitalOcean DNS: https://cloud.digitalocean.com/networking/domains
- If using another DNS provider: Login to your domain registrar

## Step 6: Verify Deployment

After DNS propagates (can take 5-60 minutes):

1. **Visit**: http://video-compression.iffuso.com
2. **Or visit directly**: http://$DEPLOY_SERVER_IP

You should see the PF-Compression application!

## Optional: Enable HTTPS (Recommended)

Once DNS is working, enable HTTPS:

```bash
ssh root@$DEPLOY_SERVER_IP

# Install Certbot
apt-get update
apt-get install -y certbot python3-certbot-nginx

# Get SSL certificate
certbot certonly --standalone -d video-compression.iffuso.com --pre-hook "docker-compose -f /opt/pf-compression/docker-compose.prod.yml down" --post-hook "docker-compose -f /opt/pf-compression/docker-compose.prod.yml up -d"
```

Then update the nginx configuration to use SSL.

## Troubleshooting

### SSH Connection Fails
```powershell
# Check if SSH key is correct
Get-Content "$env:USERPROFILE\.ssh\id_rsa.pub"

# Try with verbose output
ssh -v root@$DEPLOY_SERVER_IP
```

### Deployment Fails
```powershell
# View logs
ssh root@$DEPLOY_SERVER_IP 'docker-compose -f /opt/pf-compression/docker-compose.prod.yml logs'

# Restart deployment
.\deploy\quick-deploy.ps1 -DeployOnly
```

### DNS Not Working
```powershell
# Check DNS propagation
nslookup video-compression.iffuso.com

# Use IP address directly
Start-Process "http://$DEPLOY_SERVER_IP"
```

## Quick Commands

```powershell
# View application logs
ssh root@$DEPLOY_SERVER_IP 'docker-compose -f /opt/pf-compression/docker-compose.prod.yml logs -f'

# Restart application
ssh root@$DEPLOY_SERVER_IP 'docker-compose -f /opt/pf-compression/docker-compose.prod.yml restart'

# Redeploy application
.\deploy\quick-deploy.ps1 -DeployOnly

# SSH into server
ssh root@$DEPLOY_SERVER_IP
```

## Summary

1. ✅ Get SSH public key
2. ✅ Add to droplet via console
3. ✅ Test SSH connection
4. ✅ Run `.\deploy\quick-deploy.ps1`
5. ✅ Configure DNS A record
6. ✅ Visit http://video-compression.iffuso.com

That's it! 🎉

