# Consolidated Deployment Script for PF-Compression
# Builds Docker image locally and deploys to DigitalOcean droplet
# Optionally sets up nginx reverse proxy and SSL

param(
    [switch]$SetupNginx,
    [switch]$SetupSSL,
    [switch]$SkipBuild,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Show help
if ($Help) {
    Write-Host "PF-Compression Deployment Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\deploy.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -SetupNginx    Setup nginx reverse proxy on the server" -ForegroundColor White
    Write-Host "  -SetupSSL      Setup SSL certificate with Let's Encrypt" -ForegroundColor White
    Write-Host "  -SkipBuild     Skip Docker build (use existing image)" -ForegroundColor White
    Write-Host "  -Help          Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Environment Variables (set in .env):" -ForegroundColor Yellow
    Write-Host "  DEPLOY_SERVER_IP    - Server IP address" -ForegroundColor White
    Write-Host "  TARGET_DOMAIN       - Domain name (optional)" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\deploy.ps1                    # Deploy only" -ForegroundColor White
    Write-Host "  .\deploy.ps1 -SetupNginx        # Deploy and setup nginx" -ForegroundColor White
    Write-Host "  .\deploy.ps1 -SetupNginx -SetupSSL  # Full setup with SSL" -ForegroundColor White
    Write-Host "  .\deploy.ps1 -SkipBuild         # Deploy without rebuilding" -ForegroundColor White
    exit 0
}

# Validate environment variables
$ServerIP = $env:DEPLOY_SERVER_IP
if ([string]::IsNullOrEmpty($ServerIP)) {
    Write-Host "ERROR: DEPLOY_SERVER_IP environment variable not set!" -ForegroundColor Red
    Write-Host "Please set it in your .env file or environment" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Example .env file:" -ForegroundColor Cyan
    Write-Host "  DEPLOY_SERVER_IP=your.server.ip.address" -ForegroundColor White
    Write-Host "  TARGET_DOMAIN=your-domain.com" -ForegroundColor White
    exit 1
}

$Domain = $env:TARGET_DOMAIN
$ServerUser = "root"
$SSHKey = "$env:USERPROFILE\.ssh\a-icon-deploy"
$ImageName = "pf-compression-web"
$ImageTag = "latest"
$ContainerName = "pf-compression-web"
$ContainerPort = 8080

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         PF-Compression Deployment Script                  ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "Server IP:    $ServerIP" -ForegroundColor White
if ($Domain) {
    Write-Host "Domain:       $Domain" -ForegroundColor White
}
Write-Host "Setup Nginx:  $SetupNginx" -ForegroundColor White
Write-Host "Setup SSL:    $SetupSSL" -ForegroundColor White
Write-Host "Skip Build:   $SkipBuild" -ForegroundColor White
Write-Host ""

# Test SSH connection
Write-Host "Testing SSH connection..." -ForegroundColor Yellow
try {
    ssh -i $SSHKey -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${ServerUser}@${ServerIP} "echo 'Connected'" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "SSH connection failed"
    }
    Write-Host "✓ SSH connection successful" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "ERROR: Cannot connect to server!" -ForegroundColor Red
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "  1. Server IP is correct" -ForegroundColor White
    Write-Host "  2. SSH key exists at: $SSHKey" -ForegroundColor White
    Write-Host "  3. Server is accessible" -ForegroundColor White
    exit 1
}

# Step 1: Build Docker image locally
if (-not $SkipBuild) {
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host " Step 1/4: Building Docker Image" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    docker build -t ${ImageName}:${ImageTag} .
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    Write-Host "✓ Docker image built successfully" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "Skipping Docker build (using existing image)..." -ForegroundColor Yellow
    Write-Host ""
}

# Step 2: Save image to tar file
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " Step 2/4: Packaging Docker Image" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$TempFile = "$env:TEMP\pf-compression-image.tar"
Write-Host "Saving image to: $TempFile" -ForegroundColor White
docker save -o $TempFile ${ImageName}:${ImageTag}
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to save Docker image!" -ForegroundColor Red
    exit 1
}

$FileSize = (Get-Item $TempFile).Length / 1MB
Write-Host "✓ Image saved: $([math]::Round($FileSize, 2)) MB" -ForegroundColor Green
Write-Host ""

# Step 3: Upload image to server
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " Step 3/4: Uploading to Server" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Write-Host "Uploading image to ${ServerIP}..." -ForegroundColor White
scp -i $SSHKey -o StrictHostKeyChecking=no -o Compression=yes $TempFile ${ServerUser}@${ServerIP}:/tmp/pf-compression-image.tar
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to upload image!" -ForegroundColor Red
    Remove-Item $TempFile -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "✓ Upload complete" -ForegroundColor Green
Write-Host ""

# Step 4: Deploy on server
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " Step 4/4: Deploying Container" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$DeployCommands = @"
echo "Stopping existing container..."
docker stop $ContainerName 2>/dev/null || true
docker rm $ContainerName 2>/dev/null || true

echo "Loading new image..."
docker load -i /tmp/pf-compression-image.tar

echo "Removing temporary file..."
rm /tmp/pf-compression-image.tar

echo "Starting container..."
docker run -d --name $ContainerName --restart unless-stopped -p ${ContainerPort}:80 ${ImageName}:${ImageTag}

echo ""
echo "Container status:"
docker ps | grep $ContainerName
"@

ssh -i $SSHKey ${ServerUser}@${ServerIP} $DeployCommands
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to deploy container!" -ForegroundColor Red
    Remove-Item $TempFile -Force -ErrorAction SilentlyContinue
    exit 1
}

# Cleanup local tar file
Remove-Item $TempFile -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "✓ Container deployed successfully" -ForegroundColor Green
Write-Host ""

# Setup Nginx if requested
if ($SetupNginx) {
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host " Setting up Nginx Reverse Proxy" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""

    if ([string]::IsNullOrEmpty($Domain)) {
        Write-Host "ERROR: TARGET_DOMAIN environment variable not set!" -ForegroundColor Red
        Write-Host "Nginx setup requires a domain name." -ForegroundColor Yellow
        exit 1
    }

    # Prepare nginx configuration
    Write-Host "Preparing nginx configuration for $Domain..." -ForegroundColor White
    $nginxConfig = Get-Content deploy/nginx-host.conf -Raw
    $nginxConfig = $nginxConfig -replace 'video-compression\.iffuso\.com', $Domain

    $tempLocalFile = "$env:TEMP\nginx-${Domain}.conf"
    $nginxConfig | Set-Content $tempLocalFile -NoNewline

    # Upload configuration
    Write-Host "Uploading nginx configuration..." -ForegroundColor White
    $TempConfName = $Domain -replace '\.', '-'
    scp -i $SSHKey -o StrictHostKeyChecking=no $tempLocalFile ${ServerUser}@${ServerIP}:/tmp/${TempConfName}.conf

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to upload nginx config!" -ForegroundColor Red
        Remove-Item $tempLocalFile -Force -ErrorAction SilentlyContinue
        exit 1
    }

    Remove-Item $tempLocalFile -Force -ErrorAction SilentlyContinue

    # Install and configure nginx
    Write-Host "Configuring nginx on server..." -ForegroundColor White

    $NginxCommands = @"
# Install nginx if not already installed
if ! command -v nginx &> /dev/null; then
    apt-get update
    apt-get install -y nginx
fi

# Copy configuration
cp /tmp/${TempConfName}.conf /etc/nginx/sites-available/${Domain}

# Enable site
ln -sf /etc/nginx/sites-available/${Domain} /etc/nginx/sites-enabled/

# Test nginx configuration
nginx -t

# Reload nginx
systemctl reload nginx

# Ensure nginx is enabled and running
systemctl enable nginx
systemctl start nginx

echo ""
echo "Nginx configured successfully!"
"@

    ssh -i $SSHKey ${ServerUser}@${ServerIP} $NginxCommands

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to configure nginx!" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "✓ Nginx reverse proxy configured" -ForegroundColor Green
    Write-Host ""
}

# Setup SSL if requested
if ($SetupSSL) {
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host " Setting up SSL Certificate" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""

    if ([string]::IsNullOrEmpty($Domain)) {
        Write-Host "ERROR: TARGET_DOMAIN environment variable not set!" -ForegroundColor Red
        Write-Host "SSL setup requires a domain name." -ForegroundColor Yellow
        exit 1
    }

    # Verify DNS
    Write-Host "Verifying DNS configuration for $Domain..." -ForegroundColor White
    $DNSOutput = nslookup $Domain 2>&1 | Out-String
    $DNSResult = ($DNSOutput -split "`n" | Select-String "Address:" | Select-Object -Last 1).ToString().Trim()
    Write-Host "DNS resolves to: $DNSResult" -ForegroundColor White

    if ($DNSResult -notmatch $ServerIP) {
        Write-Host ""
        Write-Host "WARNING: DNS is not pointing to $ServerIP" -ForegroundColor Yellow
        Write-Host "SSL certificate may fail. Continue anyway? (y/N)" -ForegroundColor Yellow
        $continue = Read-Host
        if ($continue -ne "y") {
            Write-Host "Skipping SSL setup." -ForegroundColor Yellow
            exit 0
        }
    }

    # Install certbot and obtain certificate
    Write-Host "Installing certbot and obtaining SSL certificate..." -ForegroundColor White

    $SSLCommands = @"
# Install certbot if not already installed
if ! command -v certbot &> /dev/null; then
    apt-get update
    apt-get install -y certbot python3-certbot-nginx
fi

# Obtain SSL certificate
certbot --nginx -d $Domain --non-interactive --agree-tos --email admin@iffuso.com --redirect

echo ""
echo "SSL certificate obtained successfully!"
certbot certificates
"@

    ssh -i $SSHKey ${ServerUser}@${ServerIP} $SSLCommands

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Failed to obtain SSL certificate!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Common issues:" -ForegroundColor Yellow
        Write-Host "  1. DNS not propagated yet (wait 5-30 minutes)" -ForegroundColor White
        Write-Host "  2. DNS pointing to wrong IP address" -ForegroundColor White
        Write-Host "  3. Port 80 not accessible from internet" -ForegroundColor White
        exit 1
    }

    Write-Host ""
    Write-Host "✓ SSL certificate configured" -ForegroundColor Green
    Write-Host ""
}

# Final summary
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              Deployment Complete!                          ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Application URLs:" -ForegroundColor Cyan
Write-Host "  Direct IP:  http://${ServerIP}:${ContainerPort}" -ForegroundColor White

if ($Domain) {
    if ($SetupSSL) {
        Write-Host "  Domain:     https://${Domain}" -ForegroundColor White
    } else {
        Write-Host "  Domain:     http://${Domain}" -ForegroundColor White
        if ($SetupNginx) {
            Write-Host "              (Run with -SetupSSL to enable HTTPS)" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  View logs:     ssh -i $SSHKey ${ServerUser}@${ServerIP} 'docker logs -f $ContainerName'" -ForegroundColor White
Write-Host "  Restart:       ssh -i $SSHKey ${ServerUser}@${ServerIP} 'docker restart $ContainerName'" -ForegroundColor White
Write-Host "  Stop:          ssh -i $SSHKey ${ServerUser}@${ServerIP} 'docker stop $ContainerName'" -ForegroundColor White
Write-Host ""


