# Setup SSL certificate for video-compression.iffuso.com
$ErrorActionPreference = "Stop"

$ServerIP = "167.71.191.234"
$ServerUser = "root"
$SSHKey = "$env:USERPROFILE\.ssh\a-icon-deploy"
$Domain = "video-compression.iffuso.com"

Write-Host "=== SSL Certificate Setup for $Domain ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Verify DNS
Write-Host "[1/3] Verifying DNS configuration..." -ForegroundColor Yellow
$DNSOutput = nslookup $Domain 2>&1 | Out-String
$DNSResult = ($DNSOutput -split "`n" | Select-String "Address:" | Select-Object -Last 1).ToString().Trim()
Write-Host "Current DNS: $DNSResult" -ForegroundColor White

if ($DNSResult -notmatch $ServerIP) {
    Write-Host ""
    Write-Host "WARNING: DNS is not pointing to the correct IP!" -ForegroundColor Red
    Write-Host "Expected: $ServerIP" -ForegroundColor Yellow
    Write-Host "Current:  $DNSResult" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please update your DNS A record:" -ForegroundColor Cyan
    Write-Host "  Type: A" -ForegroundColor White
    Write-Host "  Host: video-compression" -ForegroundColor White
    Write-Host "  Value: $ServerIP" -ForegroundColor White
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne "y") {
        Write-Host "Exiting. Please update DNS first." -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "DNS verification complete!" -ForegroundColor Green
Write-Host ""

# Step 2: Install certbot
Write-Host "[2/3] Installing certbot..." -ForegroundColor Yellow

$InstallCommands = @'
# Install certbot if not already installed
if ! command -v certbot &> /dev/null; then
    apt-get update
    apt-get install -y certbot python3-certbot-nginx
fi

# Show certbot version
certbot --version
'@

ssh -i $SSHKey ${ServerUser}@${ServerIP} $InstallCommands
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install certbot!" -ForegroundColor Red
    exit 1
}
Write-Host "Certbot installed!" -ForegroundColor Green
Write-Host ""

# Step 3: Obtain SSL certificate
Write-Host "[3/3] Obtaining SSL certificate..." -ForegroundColor Yellow
Write-Host "This will:" -ForegroundColor Cyan
Write-Host "  - Obtain a free SSL certificate from Let's Encrypt" -ForegroundColor White
Write-Host "  - Automatically configure nginx for HTTPS" -ForegroundColor White
Write-Host "  - Set up auto-renewal" -ForegroundColor White
Write-Host ""

$SSLCommands = @"
certbot --nginx -d $Domain --non-interactive --agree-tos --email admin@iffuso.com --redirect
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
    Write-Host ""
    Write-Host "To check DNS propagation:" -ForegroundColor Cyan
    Write-Host "  nslookup $Domain" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "=== SSL Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Your application is now accessible via HTTPS:" -ForegroundColor Cyan
Write-Host "  https://$Domain" -ForegroundColor White
Write-Host ""
Write-Host "Certificate details:" -ForegroundColor Cyan
ssh -i $SSHKey ${ServerUser}@${ServerIP} "certbot certificates"
Write-Host ""
Write-Host "Auto-renewal is configured. Certificate will renew automatically." -ForegroundColor Green
Write-Host ""

