#!/bin/bash

# PF-Compression PWA Deployment Script
# This script helps deploy the PWA to DigitalOcean

set -e

echo "🚀 PF-Compression PWA Deployment Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if required tools are installed
check_requirements() {
    print_info "Checking requirements..."
    
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform is not installed. Please install it from https://www.terraform.io/downloads"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed locally. It will be installed on the droplet."
    fi
    
    print_info "Requirements check passed!"
}

# Build the PWA locally
build_pwa() {
    print_info "Building PWA locally..."
    
    cd app/ts-pwalib
    npm install
    npm run build
    cd ../..
    
    cd app/pwa
    npm install
    npm run build
    cd ../..
    
    print_info "PWA built successfully!"
}

# Deploy with Terraform
deploy_terraform() {
    print_info "Deploying infrastructure with Terraform..."
    
    cd terraform
    
    if [ ! -f "terraform.tfvars" ]; then
        print_error "terraform.tfvars not found!"
        print_info "Please copy terraform.tfvars.example to terraform.tfvars and fill in your values."
        exit 1
    fi
    
    terraform init
    terraform plan
    
    read -p "Do you want to apply this plan? (yes/no): " confirm
    if [ "$confirm" == "yes" ]; then
        terraform apply -auto-approve
        
        DROPLET_IP=$(terraform output -raw droplet_ip)
        print_info "Droplet created at IP: $DROPLET_IP"
        
        cd ..
        
        # Wait for droplet to be ready
        print_info "Waiting for droplet to be ready (60 seconds)..."
        sleep 60
        
        return 0
    else
        print_warning "Deployment cancelled."
        exit 0
    fi
}

# Deploy application to droplet
deploy_app() {
    print_info "Deploying application to droplet..."
    
    cd terraform
    DROPLET_IP=$(terraform output -raw droplet_ip)
    cd ..
    
    # Copy files to droplet
    print_info "Copying files to droplet..."
    scp -r docker root@$DROPLET_IP:/opt/pf-compression/
    scp -r dist/pwa root@$DROPLET_IP:/opt/pf-compression/
    
    # Deploy with Docker
    print_info "Starting Docker containers..."
    ssh root@$DROPLET_IP "cd /opt/pf-compression/docker && docker-compose -f docker-compose.pwa.yml up -d --build"
    
    print_info "Application deployed successfully!"
}

# Setup SSL
setup_ssl() {
    print_info "Setting up SSL certificate..."
    
    cd terraform
    DROPLET_IP=$(terraform output -raw droplet_ip)
    DOMAIN=$(terraform output -raw domain)
    cd ..
    
    ssh root@$DROPLET_IP "certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN"
    
    print_info "SSL certificate installed!"
}

# Main menu
show_menu() {
    echo ""
    echo "What would you like to do?"
    echo "1) Full deployment (build + infrastructure + app + SSL)"
    echo "2) Build PWA only"
    echo "3) Deploy infrastructure only (Terraform)"
    echo "4) Deploy application only"
    echo "5) Setup SSL only"
    echo "6) Exit"
    echo ""
    read -p "Enter your choice [1-6]: " choice
    
    case $choice in
        1)
            check_requirements
            build_pwa
            deploy_terraform
            deploy_app
            setup_ssl
            print_info "✅ Full deployment complete!"
            if [ -n "$TARGET_DOMAIN" ]; then
                print_info "Visit https://${TARGET_DOMAIN}"
            fi
            ;;
        2)
            build_pwa
            ;;
        3)
            check_requirements
            deploy_terraform
            ;;
        4)
            deploy_app
            ;;
        5)
            setup_ssl
            ;;
        6)
            print_info "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid choice!"
            show_menu
            ;;
    esac
}

# Run the menu
show_menu

