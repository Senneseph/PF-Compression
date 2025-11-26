# Simple deployment script for PF-Compression
param(
    [string]$ServerIP = $env:DEPLOY_SERVER_IP,
    [string]$ServerUser = "root"
)

if ([string]::IsNullOrEmpty($ServerIP)) {
    Write-Host "ERROR: DEPLOY_SERVER_IP environment variable not set!" -ForegroundColor Red
    Write-Host "Please set it in your .env file or environment, or pass -ServerIP parameter" -ForegroundColor Yellow
    exit 1
}

$ErrorActionPreference = "Stop"
$SSHKey = "$env:USERPROFILE\.ssh\a-icon-deploy"
$AppDir = "/opt/pf-compression"

Write-Host "Deploying PF-Compression to production..." -ForegroundColor Cyan
Write-Host ""

# Test connection
Write-Host "Testing SSH connection..." -ForegroundColor Yellow
ssh -i $SSHKey "${ServerUser}@${ServerIP}" "echo 'Connected'"

# Create temp directory
Write-Host "Preparing deployment package..." -ForegroundColor Yellow
$TempDir = New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -ItemType Directory -Path $_ }

try {
    # Copy files
    Copy-Item -Path "app" -Destination $TempDir -Recurse
    Copy-Item -Path "Dockerfile" -Destination $TempDir
    Copy-Item -Path "docker-compose.prod.yml" -Destination $TempDir
    New-Item -ItemType Directory -Path "$TempDir/deploy" -Force | Out-Null
    Copy-Item -Path "deploy/nginx.conf" -Destination "$TempDir/deploy/nginx.conf"
    
    # Upload files
    Write-Host "Uploading files to server..." -ForegroundColor Yellow
    scp -i $SSHKey -r -o StrictHostKeyChecking=no "$TempDir\*" "${ServerUser}@${ServerIP}:${AppDir}/"
    
    # Stop existing containers
    Write-Host "Stopping existing containers..." -ForegroundColor Yellow
    ssh -i $SSHKey "${ServerUser}@${ServerIP}" "cd ${AppDir}; docker-compose -f docker-compose.prod.yml down" 2>$null
    
    # Build image
    Write-Host "Building Docker image (this may take a few minutes)..." -ForegroundColor Yellow
    ssh -i $SSHKey "${ServerUser}@${ServerIP}" "cd ${AppDir}; docker-compose -f docker-compose.prod.yml build --no-cache"
    
    # Start containers
    Write-Host "Starting containers..." -ForegroundColor Yellow
    ssh -i $SSHKey "${ServerUser}@${ServerIP}" "cd ${AppDir}; docker-compose -f docker-compose.prod.yml up -d"
    
    # Wait for startup
    Write-Host "Waiting for application to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    
    # Check status
    $status = ssh -i $SSHKey "${ServerUser}@${ServerIP}" "docker-compose -f ${AppDir}/docker-compose.prod.yml ps"
    
    if ($status -match "Up") {
        Write-Host ""
        Write-Host "Deployment successful!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Application URLs:" -ForegroundColor Cyan
        Write-Host "  http://video-compression.iffuso.com (configure DNS A record to ${ServerIP})" -ForegroundColor White
        Write-Host "  http://${ServerIP}" -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host "Container not running. Checking logs..." -ForegroundColor Red
        ssh -i $SSHKey "${ServerUser}@${ServerIP}" "docker-compose -f ${AppDir}/docker-compose.prod.yml logs --tail=50"
    }
    
} finally {
    # Cleanup
    Remove-Item -Path $TempDir -Recurse -Force
}

