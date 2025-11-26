# Fast deployment script - builds locally, pushes image to server
# This avoids uploading all source files and node_modules

$ErrorActionPreference = "Stop"

$ServerIP = $env:DEPLOY_SERVER_IP
if ([string]::IsNullOrEmpty($ServerIP)) {
    Write-Host "ERROR: DEPLOY_SERVER_IP environment variable not set!" -ForegroundColor Red
    Write-Host "Please set it in your .env file or environment" -ForegroundColor Yellow
    exit 1
}

$ServerUser = "root"
$SSHKey = "$env:USERPROFILE\.ssh\a-icon-deploy"
$ImageName = "pf-compression-web"
$ImageTag = "latest"

Write-Host "=== Fast Deployment to $ServerIP ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Build Docker image locally
Write-Host "[1/4] Building Docker image locally..." -ForegroundColor Yellow
docker build -t ${ImageName}:${ImageTag} .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Build complete!" -ForegroundColor Green
Write-Host ""

# Step 2: Save image to tar file
Write-Host "[2/4] Saving Docker image to tar file..." -ForegroundColor Yellow
$TempFile = "$env:TEMP\pf-compression-image.tar"
docker save -o $TempFile ${ImageName}:${ImageTag}
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to save Docker image!" -ForegroundColor Red
    exit 1
}
$FileSize = (Get-Item $TempFile).Length / 1MB
Write-Host "Image saved! Size: $([math]::Round($FileSize, 2)) MB" -ForegroundColor Green
Write-Host ""

# Step 3: Upload image to server
Write-Host "[3/4] Uploading image to server..." -ForegroundColor Yellow
scp -i $SSHKey -o StrictHostKeyChecking=no -o Compression=yes $TempFile ${ServerUser}@${ServerIP}:/tmp/pf-compression-image.tar
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to upload image!" -ForegroundColor Red
    Remove-Item $TempFile -Force
    exit 1
}
Write-Host "Upload complete!" -ForegroundColor Green
Write-Host ""

# Step 4: Load and run on server
Write-Host "[4/4] Loading image and starting container on server..." -ForegroundColor Yellow

$RemoteCommands = @'
# Stop and remove existing container
docker stop pf-compression-web 2>/dev/null || true
docker rm pf-compression-web 2>/dev/null || true

# Load the new image
docker load -i /tmp/pf-compression-image.tar

# Remove the tar file
rm /tmp/pf-compression-image.tar

# Run the container on port 8080
docker run -d --name pf-compression-web --restart unless-stopped -p 8080:80 pf-compression-web:latest

# Show status
echo ""
echo "Container status:"
docker ps | grep pf-compression-web
'@

ssh -i $SSHKey ${ServerUser}@${ServerIP} $RemoteCommands
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to start container on server!" -ForegroundColor Red
    Remove-Item $TempFile -Force
    exit 1
}

# Cleanup local tar file
Remove-Item $TempFile -Force

Write-Host ""
Write-Host "=== Deployment Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Application is now running at:" -ForegroundColor Cyan
Write-Host "  http://$ServerIP:8080 (direct access)" -ForegroundColor White
if ($env:TARGET_DOMAIN) {
    Write-Host "  http://$env:TARGET_DOMAIN (once DNS is configured)" -ForegroundColor White
}
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Cyan
Write-Host "  ssh -i $SSHKey ${ServerUser}@${ServerIP} 'docker logs -f pf-compression-web'" -ForegroundColor White
Write-Host ""

