# PowerShell deployment script for PF-Compression
# Deploys directly from local machine to DigitalOcean droplet

param(
    [string]$ServerIP = "167.71.191.234",
    [string]$ServerUser = "root"
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Deploying PF-Compression to production..." -ForegroundColor Cyan

# Configuration
$AppDir = "/opt/pf-compression"
$LocalPath = $PSScriptRoot + "\.."

# Test SSH connection
Write-Host "🔌 Testing connection to server..." -ForegroundColor Yellow
try {
    ssh -o StrictHostKeyChecking=no "${ServerUser}@${ServerIP}" "echo 'Connection successful'"
} catch {
    Write-Host "❌ Failed to connect to server. Please check your SSH configuration." -ForegroundColor Red
    exit 1
}

# Create app directory on server
Write-Host "📁 Creating application directory..." -ForegroundColor Yellow
ssh "${ServerUser}@${ServerIP}" "mkdir -p ${AppDir}"

# Copy files to server (excluding node_modules, dist, etc.)
Write-Host "📤 Copying files to server..." -ForegroundColor Yellow
scp -r `
    -o StrictHostKeyChecking=no `
    "${LocalPath}\app" `
    "${LocalPath}\Dockerfile" `
    "${LocalPath}\docker-compose.prod.yml" `
    "${LocalPath}\deploy" `
    "${ServerUser}@${ServerIP}:${AppDir}/"

# Build and deploy
Write-Host "🐳 Building and deploying containers..." -ForegroundColor Yellow
ssh "${ServerUser}@${ServerIP}" @"
    cd ${AppDir}
    docker-compose -f docker-compose.prod.yml down || true
    docker-compose -f docker-compose.prod.yml build --no-cache
    docker-compose -f docker-compose.prod.yml up -d
"@

# Wait for container to start
Write-Host "⏳ Waiting for container to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check deployment status
Write-Host "✅ Checking deployment status..." -ForegroundColor Yellow
$status = ssh "${ServerUser}@${ServerIP}" "docker-compose -f ${AppDir}/docker-compose.prod.yml ps"

if ($status -match "Up") {
    Write-Host "✅ Deployment successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Application is now running at:" -ForegroundColor Cyan
    Write-Host "   http://video-compression.iffuso.com" -ForegroundColor White
    Write-Host "   http://${ServerIP}" -ForegroundColor White
} else {
    Write-Host "❌ Deployment failed. Checking logs..." -ForegroundColor Red
    ssh "${ServerUser}@${ServerIP}" "docker-compose -f ${AppDir}/docker-compose.prod.yml logs --tail=50"
    exit 1
}

# Show recent logs
Write-Host ""
Write-Host "📋 Recent logs:" -ForegroundColor Yellow
ssh "${ServerUser}@${ServerIP}" "docker-compose -f ${AppDir}/docker-compose.prod.yml logs --tail=20"

Write-Host ""
Write-Host "✨ Deployment complete!" -ForegroundColor Green

