terraform {
  required_version = ">= 1.0"
  
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

# Configure the DigitalOcean Provider
provider "digitalocean" {
  token = var.do_token
}

# Create a new SSH key
resource "digitalocean_ssh_key" "default" {
  name       = "pf-compression-key"
  public_key = file(var.ssh_public_key_path)
}

# Create a new Droplet
resource "digitalocean_droplet" "pwa" {
  image    = "docker-20-04"
  name     = "pf-compression-pwa"
  region   = var.region
  size     = var.droplet_size
  ssh_keys = [digitalocean_ssh_key.default.fingerprint]
  
  # User data script to set up the droplet
  user_data = templatefile("${path.module}/user-data.sh", {
    domain = var.domain
  })
  
  tags = ["pf-compression", "pwa", "production"]
}

# Create a firewall
resource "digitalocean_firewall" "pwa" {
  name = "pf-compression-pwa-firewall"
  
  droplet_ids = [digitalocean_droplet.pwa.id]
  
  # Allow SSH
  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  # Allow HTTP
  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  # Allow HTTPS
  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  # Allow all outbound traffic
  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}

# DNS A record for the domain
resource "digitalocean_record" "pwa" {
  domain = var.base_domain
  type   = "A"
  name   = var.subdomain
  value  = digitalocean_droplet.pwa.ipv4_address
  ttl    = 300
}

# Output the droplet's IP address
output "droplet_ip" {
  value       = digitalocean_droplet.pwa.ipv4_address
  description = "The public IP address of the droplet"
}

output "domain" {
  value       = "${var.subdomain}.${var.base_domain}"
  description = "The full domain name"
}

