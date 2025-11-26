# Setup SSH access and deploy using DigitalOcean API
# This script will add your SSH key to the droplet and then deploy

param(
    [string]$DropletID = "530300735",
    [string]$DOToken = $env:DIGITALOCEAN_ACCESS_TOKEN
)

if ([string]::IsNullOrEmpty($DOToken)) {
    Write-Host "ERROR: DigitalOcean token not found!" -ForegroundColor Red
    Write-Host "Please set the DIGITALOCEAN_ACCESS_TOKEN environment variable or pass it via -DOToken parameter" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Example:" -ForegroundColor Cyan
    Write-Host '  $env:DIGITALOCEAN_ACCESS_TOKEN = "your_token_here"' -ForegroundColor White
    Write-Host "  .\deploy\setup-ssh-and-deploy.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Or:" -ForegroundColor Cyan
    Write-Host '  .\deploy\setup-ssh-and-deploy.ps1 -DOToken "your_token_here"' -ForegroundColor White
    exit 1
}

$ErrorActionPreference = "Stop"

Write-Host "🔑 Setting up SSH access to DigitalOcean droplet..." -ForegroundColor Cyan

# Check if SSH key exists
$SSHKeyPath = "$env:USERPROFILE\.ssh\id_rsa.pub"
if (-not (Test-Path $SSHKeyPath)) {
    Write-Host "📝 No SSH key found. Generating new SSH key..." -ForegroundColor Yellow
    ssh-keygen -t rsa -b 4096 -f "$env:USERPROFILE\.ssh\id_rsa" -N '""'
}

# Read SSH public key
$SSHPublicKey = Get-Content $SSHKeyPath -Raw
$SSHPublicKey = $SSHPublicKey.Trim()

Write-Host "📤 Adding SSH key to DigitalOcean..." -ForegroundColor Yellow

# Add SSH key to DigitalOcean account
$headers = @{
    "Authorization" = "Bearer $DOToken"
    "Content-Type" = "application/json"
}

$keyName = "deployment-key-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$body = @{
    name = $keyName
    public_key = $SSHPublicKey
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "https://api.digitalocean.com/v2/account/keys" -Method Post -Headers $headers -Body $body
    $sshKeyID = $response.ssh_key.id
    Write-Host "✅ SSH key added to DigitalOcean (ID: $sshKeyID)" -ForegroundColor Green
} catch {
    if ($_.Exception.Response.StatusCode -eq 422) {
        Write-Host "⚠️  SSH key already exists in DigitalOcean" -ForegroundColor Yellow
        # Get existing keys and find ours
        $keys = Invoke-RestMethod -Uri "https://api.digitalocean.com/v2/account/keys" -Headers $headers
        $existingKey = $keys.ssh_keys | Where-Object { $_.public_key -eq $SSHPublicKey } | Select-Object -First 1
        if ($existingKey) {
            $sshKeyID = $existingKey.id
            Write-Host "✅ Using existing SSH key (ID: $sshKeyID)" -ForegroundColor Green
        }
    } else {
        throw
    }
}

# Add SSH key to droplet
Write-Host "🔧 Adding SSH key to droplet..." -ForegroundColor Yellow

# First, we need to use the DigitalOcean console or recovery mode to add the key
# For now, let's try to enable password authentication temporarily

Write-Host ""
Write-Host "⚠️  Manual step required:" -ForegroundColor Yellow
Write-Host "To complete SSH setup, you need to either:" -ForegroundColor White
Write-Host ""
Write-Host "Option 1: Use DigitalOcean Console" -ForegroundColor Cyan
Write-Host "  1. Go to: https://cloud.digitalocean.com/droplets/$DropletID/console" -ForegroundColor White
Write-Host "  2. Login as root" -ForegroundColor White
Write-Host "  3. Run: mkdir -p ~/.ssh && echo '$SSHPublicKey' >> ~/.ssh/authorized_keys" -ForegroundColor White
Write-Host "  4. Run: chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys" -ForegroundColor White
Write-Host ""
Write-Host "Option 2: Use DigitalOcean API to rebuild droplet with SSH key" -ForegroundColor Cyan
Write-Host "  (This will erase all data on the droplet)" -ForegroundColor Red
Write-Host ""

$choice = Read-Host "Would you like to rebuild the droplet with the SSH key? (yes/no)"

if ($choice -eq "yes") {
    Write-Host "🔄 Rebuilding droplet with SSH key..." -ForegroundColor Yellow
    
    $rebuildBody = @{
        type = "rebuild"
        image = "ubuntu-24-04-x64"
    } | ConvertTo-Json
    
    try {
        $action = Invoke-RestMethod -Uri "https://api.digitalocean.com/v2/droplets/$DropletID/actions" -Method Post -Headers $headers -Body $rebuildBody
        Write-Host "✅ Rebuild initiated. Waiting for completion..." -ForegroundColor Yellow
        
        # Wait for rebuild to complete
        $actionID = $action.action.id
        do {
            Start-Sleep -Seconds 10
            $actionStatus = Invoke-RestMethod -Uri "https://api.digitalocean.com/v2/actions/$actionID" -Headers $headers
            Write-Host "   Status: $($actionStatus.action.status)" -ForegroundColor Gray
        } while ($actionStatus.action.status -eq "in-progress")
        
        if ($actionStatus.action.status -eq "completed") {
            Write-Host "✅ Droplet rebuilt successfully!" -ForegroundColor Green
            Write-Host "⏳ Waiting for droplet to boot..." -ForegroundColor Yellow
            Start-Sleep -Seconds 30
            
            # Now run the deployment
            Write-Host ""
            & "$PSScriptRoot\quick-deploy.ps1"
        } else {
            Write-Host "❌ Rebuild failed: $($actionStatus.action.status)" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Error rebuilding droplet: $_" -ForegroundColor Red
    }
} else {
    Write-Host ""
    Write-Host "Please add the SSH key manually using the DigitalOcean console, then run:" -ForegroundColor Yellow
    Write-Host "  .\deploy\quick-deploy.ps1" -ForegroundColor White
}

