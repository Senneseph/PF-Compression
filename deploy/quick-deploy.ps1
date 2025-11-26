# Quick deployment script - sets up server and deploys in one go
# Run this from the project root: .\deploy\quick-deploy.ps1

param(
    [string]$ServerIP = "167.71.191.234",
    [string]$ServerUser = "root",
    [switch]$SetupOnly,
    [switch]$DeployOnly
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 PF-Compression Deployment Script" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$AppDir = "/opt/pf-compression"

# SSH key path
$SSHKey = "$env:USERPROFILE\.ssh\a-icon-deploy"

# Test SSH connection
Write-Host "🔌 Testing SSH connection to ${ServerIP}..." -ForegroundColor Yellow
try {
    $result = ssh -i $SSHKey -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${ServerUser}@${ServerIP}" "echo 'OK'"
    if ($result -ne "OK") {
        throw "Connection test failed"
    }
    Write-Host "✅ SSH connection successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Cannot connect to server. Please ensure:" -ForegroundColor Red
    Write-Host "   1. SSH is installed (try: winget install OpenSSH.Client)" -ForegroundColor Yellow
    Write-Host "   2. You have SSH key access to the server" -ForegroundColor Yellow
    Write-Host "   3. The server IP is correct: ${ServerIP}" -ForegroundColor Yellow
    exit 1
}

# Setup server (if not deploy-only)
if (-not $DeployOnly) {
    Write-Host ""
    Write-Host "📦 Setting up server..." -ForegroundColor Cyan

    ssh -i $SSHKey "${ServerUser}@${ServerIP}" @"
        set -e
        echo '📦 Updating system...'
        apt-get update -qq
        
        echo '🐳 Installing Docker...'
        if ! command -v docker &> /dev/null; then
            curl -fsSL https://get.docker.com -o get-docker.sh
            sh get-docker.sh
            rm get-docker.sh
        fi
        
        echo '🐳 Installing Docker Compose...'
        if ! command -v docker-compose &> /dev/null; then
            curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
            chmod +x /usr/local/bin/docker-compose
        fi
        
        echo '🔥 Configuring firewall...'
        ufw allow 22/tcp
        ufw allow 80/tcp
        ufw allow 443/tcp
        ufw --force enable || true
        
        echo '📁 Creating app directory...'
        mkdir -p ${AppDir}
        
        echo '✅ Server setup complete!'
"@
    
    if ($SetupOnly) {
        Write-Host ""
        Write-Host "✅ Server setup complete! Run with -DeployOnly to deploy the app." -ForegroundColor Green
        exit 0
    }
}

# Deploy application
if (-not $SetupOnly) {
    Write-Host ""
    Write-Host "📤 Deploying application..." -ForegroundColor Cyan
    
    # Create temporary directory for deployment
    $TempDir = New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -ItemType Directory -Path $_ }
    
    try {
        # Copy necessary files to temp directory
        Write-Host "📋 Preparing deployment package..." -ForegroundColor Yellow
        Copy-Item -Path "app" -Destination $TempDir -Recurse
        Copy-Item -Path "Dockerfile" -Destination $TempDir
        Copy-Item -Path "docker-compose.prod.yml" -Destination $TempDir
        Copy-Item -Path "deploy/nginx.conf" -Destination "$TempDir/nginx.conf"
        
        # Create deploy directory structure
        New-Item -ItemType Directory -Path "$TempDir/deploy" -Force | Out-Null
        Move-Item -Path "$TempDir/nginx.conf" -Destination "$TempDir/deploy/nginx.conf"
        
        # Upload to server
        Write-Host "📤 Uploading files to server..." -ForegroundColor Yellow
        scp -i $SSHKey -r -o StrictHostKeyChecking=no "$TempDir/*" "${ServerUser}@${ServerIP}:${AppDir}/"

        # Stop existing containers
        Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
        ssh -i $SSHKey "${ServerUser}@${ServerIP}" "cd ${AppDir}; docker-compose -f docker-compose.prod.yml down" 2>$null

        # Build and start containers
        Write-Host "🐳 Building Docker image..." -ForegroundColor Yellow
        ssh -i $SSHKey "${ServerUser}@${ServerIP}" "cd ${AppDir}; docker-compose -f docker-compose.prod.yml build --no-cache"

        Write-Host "🚀 Starting containers..." -ForegroundColor Yellow
        ssh -i $SSHKey "${ServerUser}@${ServerIP}" "cd ${AppDir}; docker-compose -f docker-compose.prod.yml up -d"
        
        # Wait for startup
        Write-Host "⏳ Waiting for application to start..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        
        # Check status
        $status = ssh -i $SSHKey "${ServerUser}@${ServerIP}" "docker-compose -f ${AppDir}/docker-compose.prod.yml ps"
        
        if ($status -match "Up") {
            Write-Host ""
            Write-Host "✅ Deployment successful!" -ForegroundColor Green
            Write-Host ""
            Write-Host "🌐 Application URLs:" -ForegroundColor Cyan
            Write-Host "   http://video-compression.iffuso.com (configure DNS A record to ${ServerIP})" -ForegroundColor White
            Write-Host "   http://${ServerIP}" -ForegroundColor White
            Write-Host ""
            Write-Host "📋 To view logs: ssh ${ServerUser}@${ServerIP} ""docker-compose -f ${AppDir}/docker-compose.prod.yml logs -f""" -ForegroundColor Yellow
        } else {
            Write-Host "❌ Container not running. Checking logs..." -ForegroundColor Red
            ssh -i $SSHKey "${ServerUser}@${ServerIP}" "docker-compose -f ${AppDir}/docker-compose.prod.yml logs --tail=50"
        }
        
    } finally {
        # Cleanup temp directory
        Remove-Item -Path $TempDir -Recurse -Force
    }
}

