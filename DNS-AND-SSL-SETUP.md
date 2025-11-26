# DNS and SSL Setup Guide

## Current Issue

The domain `$TARGET_DOMAIN` is currently pointing to the **wrong IP address**.

### Current DNS Configuration
- **Domain**: $TARGET_DOMAIN
- **Current IP**: 134.209.39.114 ❌ (Wrong)
- **Correct IP**: $DEPLOY_SERVER_IP ✅ (Our droplet: a-icon-app)

This is why you're getting a 404 error when accessing https://$TARGET_DOMAIN/

---

## Step 1: Update DNS Record

You need to update the DNS A record to point to the correct IP address.

### DNS Configuration Required

```
Type: A
Host: video-compression
Domain: iffuso.com
Value: $DEPLOY_SERVER_IP
TTL: 3600 (or Auto)
```

### Where to Update DNS

1. **Log in to your DNS provider** (where iffuso.com is registered)
   - This could be: DigitalOcean, Cloudflare, GoDaddy, Namecheap, etc.

2. **Navigate to DNS Management**
   - Look for "DNS", "DNS Records", or "Manage DNS"

3. **Find the existing A record**
   - Look for: `$TARGET_DOMAIN` or `video-compression`
   - Current value: `134.209.39.114`

4. **Update the A record**
   - Change the IP address to: `$DEPLOY_SERVER_IP`
   - Save the changes

5. **Wait for DNS propagation**
   - Usually takes 5-30 minutes
   - Can take up to 48 hours in rare cases

### Verify DNS Update

Run this command to check if DNS has updated:

```powershell
nslookup $TARGET_DOMAIN
```

You should see:
```
Address: $DEPLOY_SERVER_IP
```

---

## Step 2: Test HTTP Access

Once DNS is updated, test HTTP access:

```powershell
# Test from command line
Invoke-WebRequest -Uri "http://$TARGET_DOMAIN" -UseBasicParsing

# Or open in browser
start http://$TARGET_DOMAIN
```

You should see the PF-Compression application!

---

## Step 3: Enable SSL/HTTPS

Once DNS is correctly configured and HTTP is working, run:

```powershell
.\deploy\setup-ssl.ps1
```

This will:
- ✅ Verify DNS is pointing to the correct IP
- ✅ Install certbot (if not already installed)
- ✅ Obtain a free SSL certificate from Let's Encrypt
- ✅ Automatically configure nginx for HTTPS
- ✅ Set up auto-renewal (certificate renews automatically)
- ✅ Redirect HTTP to HTTPS

### Manual SSL Setup (Alternative)

If you prefer to set up SSL manually:

```bash
ssh -i ~/.ssh/a-icon-deploy root@$DEPLOY_SERVER_IP

# Install certbot
apt-get update
apt-get install -y certbot python3-certbot-nginx

# Obtain certificate
certbot --nginx -d $TARGET_DOMAIN --non-interactive --agree-tos --email your-email@example.com --redirect
```

---

## Current Status

### ✅ Working Now
- **Direct IP Access**: http://$DEPLOY_SERVER_IP:8080
- **Container**: Running on port 8080
- **Nginx Proxy**: Configured and ready

### ⏳ Pending
- **DNS Update**: Change IP from 134.209.39.114 to $DEPLOY_SERVER_IP
- **Domain Access**: http://$TARGET_DOMAIN (after DNS update)
- **SSL/HTTPS**: https://$TARGET_DOMAIN (after DNS update)

---

## Troubleshooting

### DNS Not Updating?

1. **Check your DNS provider**
   - Make sure you're updating the correct DNS zone
   - Some providers have separate "DNS Management" sections

2. **Clear DNS cache**
   ```powershell
   ipconfig /flushdns
   ```

3. **Check DNS propagation**
   - Use online tools: https://dnschecker.org/
   - Enter: `$TARGET_DOMAIN`
   - Should show: `$DEPLOY_SERVER_IP`

### Still Getting 404?

1. **Verify the correct IP**
   ```powershell
   nslookup $TARGET_DOMAIN
   ```
   Should return: `$DEPLOY_SERVER_IP`

2. **Test direct IP access**
   ```powershell
   start http://$DEPLOY_SERVER_IP:8080
   ```
   This should work immediately

3. **Check nginx configuration**
   ```bash
   ssh -i ~/.ssh/a-icon-deploy root@$DEPLOY_SERVER_IP "nginx -t"
   ```

### SSL Certificate Fails?

Common reasons:
1. **DNS not propagated yet** - Wait longer and try again
2. **DNS pointing to wrong IP** - Verify with `nslookup`
3. **Port 80 blocked** - Check firewall rules
4. **Domain not accessible** - Test HTTP first before HTTPS

---

## Quick Reference

### Check DNS
```powershell
nslookup $TARGET_DOMAIN
```

### Test HTTP
```powershell
Invoke-WebRequest -Uri "http://$TARGET_DOMAIN" -UseBasicParsing
```

### Setup SSL
```powershell
.\deploy\setup-ssl.ps1
```

### Check Container
```powershell
ssh -i "$env:USERPROFILE\.ssh\a-icon-deploy" root@$DEPLOY_SERVER_IP "docker ps | grep pf-compression"
```

### View Logs
```powershell
ssh -i "$env:USERPROFILE\.ssh\a-icon-deploy" root@$DEPLOY_SERVER_IP "docker logs -f pf-compression-web"
```

---

## Summary

1. ✅ **Application deployed** - Running on port 8080
2. ✅ **Nginx configured** - Ready for domain traffic
3. ⏳ **Update DNS** - Change IP to $DEPLOY_SERVER_IP
4. ⏳ **Wait for propagation** - 5-30 minutes
5. ⏳ **Enable SSL** - Run `.\deploy\setup-ssl.ps1`

Once DNS is updated, your application will be accessible at:
- http://$TARGET_DOMAIN
- https://$TARGET_DOMAIN (after SSL setup)

