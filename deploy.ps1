# Consolidated Deployment Script for PF-Compression
# Builds Docker image locally and deploys to DigitalOcean droplet

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
    exit 0
}

# Load environment variables from .env file
Write-Host "Loading environment variables from .env..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
    Write-Host "[OK] Environment variables loaded" -ForegroundColor Green
} else {
    Write-Host "ERROR: .env file not found!" -ForegroundColor Red
    exit 1
}

# Configuration
$ServerIP = $env:DEPLOY_SERVER_IP
$Domain = $env:TARGET_DOMAIN
$ServerUser = "root"
$SSHKey = "$env:USERPROFILE\.ssh\a-icon-deploy"
$ImageName = "pf-compression-web"
$ImageTag = "latest"
$ContainerName = "pf-compression-web"
$ContainerPort = "8080"

if ([string]::IsNullOrEmpty($ServerIP)) {
    Write-Host "ERROR: DEPLOY_SERVER_IP environment variable not set!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Server IP:    $ServerIP" -ForegroundColor White
if ($Domain) { Write-Host "Domain:       $Domain" -ForegroundColor White }
Write-Host ""

# Test SSH connection
Write-Host "Testing SSH connection..." -ForegroundColor Yellow
ssh -i $SSHKey -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${ServerUser}@${ServerIP}" "echo Connected"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Cannot connect to server!" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] SSH connection successful" -ForegroundColor Green
Write-Host ""

# Step 1: Build Docker image locally
if (-not $SkipBuild) {
    Write-Host "Step 1/4: Building Docker image..." -ForegroundColor Cyan
    docker build -t "${ImageName}:${ImageTag}" .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Docker image built" -ForegroundColor Green
    Write-Host ""
}

# Step 2: Save image to tar file
Write-Host "Step 2/4: Saving Docker image..." -ForegroundColor Cyan
$TempFile = "$env:TEMP\pf-compression-image.tar"
docker save -o $TempFile "${ImageName}:${ImageTag}"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to save Docker image!" -ForegroundColor Red
    exit 1
}
$FileSize = [math]::Round((Get-Item $TempFile).Length / 1MB, 2)
Write-Host "[OK] Image saved: $FileSize MB" -ForegroundColor Green
Write-Host ""

# Step 3: Upload image to server
Write-Host "Step 3/4: Uploading to server..." -ForegroundColor Cyan
scp -i $SSHKey -o StrictHostKeyChecking=no -o Compression=yes $TempFile "${ServerUser}@${ServerIP}:/tmp/pf-compression-image.tar"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to upload image!" -ForegroundColor Red
    Remove-Item $TempFile -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "[OK] Upload complete" -ForegroundColor Green
Write-Host ""

# Step 4: Deploy on server
Write-Host "Step 4/4: Deploying container..." -ForegroundColor Cyan
$DeployScript = "docker stop $ContainerName 2>/dev/null; docker rm $ContainerName 2>/dev/null; docker load -i /tmp/pf-compression-image.tar; rm /tmp/pf-compression-image.tar; docker run -d --name $ContainerName --restart unless-stopped -p ${ContainerPort}:80 ${ImageName}:${ImageTag}; docker ps | grep $ContainerName"
ssh -i $SSHKey "${ServerUser}@${ServerIP}" $DeployScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to deploy container!" -ForegroundColor Red
    Remove-Item $TempFile -Force -ErrorAction SilentlyContinue
    exit 1
}
Remove-Item $TempFile -Force -ErrorAction SilentlyContinue
Write-Host "[OK] Container deployed" -ForegroundColor Green
Write-Host ""

# Setup Nginx if requested
if ($SetupNginx -and $Domain) {
    Write-Host "Setting up Nginx..." -ForegroundColor Cyan
    $nginxConfig = Get-Content deploy/nginx-host.conf -Raw
    $nginxConfig = $nginxConfig -replace 'video-compression\.iffuso\.com', $Domain
    $tempConf = "$env:TEMP\nginx-temp.conf"
    $nginxConfig | Set-Content $tempConf -NoNewline
    scp -i $SSHKey -o StrictHostKeyChecking=no $tempConf "${ServerUser}@${ServerIP}:/tmp/nginx-site.conf"
    Remove-Item $tempConf -Force -ErrorAction SilentlyContinue
    $NginxScript = "apt-get update -qq && apt-get install -y -qq nginx; cp /tmp/nginx-site.conf /etc/nginx/sites-available/$Domain; ln -sf /etc/nginx/sites-available/$Domain /etc/nginx/sites-enabled/; nginx -t && systemctl reload nginx"
    ssh -i $SSHKey "${ServerUser}@${ServerIP}" $NginxScript
    Write-Host "[OK] Nginx configured" -ForegroundColor Green
    Write-Host ""
}

# Setup SSL if requested
if ($SetupSSL -and $Domain) {
    Write-Host "Setting up SSL..." -ForegroundColor Cyan
    $SSLScript = "apt-get install -y -qq certbot python3-certbot-nginx; certbot --nginx -d $Domain --non-interactive --agree-tos --email admin@iffuso.com --redirect"
    ssh -i $SSHKey "${ServerUser}@${ServerIP}" $SSLScript
    Write-Host "[OK] SSL configured" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "  Direct IP:  http://${ServerIP}:${ContainerPort}" -ForegroundColor White
if ($Domain) {
    if ($SetupSSL) {
        Write-Host "  Domain:     https://${Domain}" -ForegroundColor White
    } else {
        Write-Host "  Domain:     http://${Domain}" -ForegroundColor White
    }
}
