# Setup nginx reverse proxy for video-compression.iffuso.com
$ErrorActionPreference = "Stop"

$ServerIP = $env:DEPLOY_SERVER_IP
if ([string]::IsNullOrEmpty($ServerIP)) {
    Write-Host "ERROR: DEPLOY_SERVER_IP environment variable not set!" -ForegroundColor Red
    Write-Host "Please set it in your .env file or environment" -ForegroundColor Yellow
    exit 1
}

$ServerUser = "root"
$SSHKey = "$env:USERPROFILE\.ssh\a-icon-deploy"

Write-Host "=== Setting up Nginx Reverse Proxy ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Upload nginx configuration
Write-Host "[1/3] Uploading nginx configuration..." -ForegroundColor Yellow
scp -i $SSHKey -o StrictHostKeyChecking=no deploy/nginx-host.conf ${ServerUser}@${ServerIP}:/tmp/video-compression.conf
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to upload nginx config!" -ForegroundColor Red
    exit 1
}
Write-Host "Upload complete!" -ForegroundColor Green
Write-Host ""

# Step 2: Install and configure nginx on host
Write-Host "[2/3] Installing and configuring nginx..." -ForegroundColor Yellow

$RemoteCommands = @'
# Install nginx if not already installed
if ! command -v nginx &> /dev/null; then
    apt-get update
    apt-get install -y nginx
fi

# Copy configuration
cp /tmp/video-compression.conf /etc/nginx/sites-available/video-compression.iffuso.com

# Enable site
ln -sf /etc/nginx/sites-available/video-compression.iffuso.com /etc/nginx/sites-enabled/

# Test nginx configuration
nginx -t

# Reload nginx
systemctl reload nginx

# Ensure nginx is enabled and running
systemctl enable nginx
systemctl start nginx

# Show status
systemctl status nginx --no-pager -l
'@

ssh -i $SSHKey ${ServerUser}@${ServerIP} $RemoteCommands
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to configure nginx!" -ForegroundColor Red
    exit 1
}
Write-Host "Nginx configured!" -ForegroundColor Green
Write-Host ""

# Step 3: Verify
Write-Host "[3/3] Verifying configuration..." -ForegroundColor Yellow
ssh -i $SSHKey ${ServerUser}@${ServerIP} "curl -s -o /dev/null -w '%{http_code}' http://localhost:8080"
$StatusCode = $LASTEXITCODE
Write-Host ""

Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Your application is now accessible at:" -ForegroundColor Cyan
Write-Host "  http://video-compression.iffuso.com" -ForegroundColor White
Write-Host "  http://$ServerIP:8080 (direct access)" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Configure DNS A record for video-compression.iffuso.com -> $ServerIP" -ForegroundColor White
Write-Host "  2. (Optional) Set up SSL with Let's Encrypt:" -ForegroundColor White
Write-Host "     certbot --nginx -d video-compression.iffuso.com" -ForegroundColor Gray
Write-Host ""

