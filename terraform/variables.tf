variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
  default     = "nyc3"
}

variable "droplet_size" {
  description = "Droplet size"
  type        = string
  default     = "s-1vcpu-1gb"
}

variable "base_domain" {
  description = "Base domain (e.g., iffuso.com)"
  type        = string
  default     = "iffuso.com"
}

variable "subdomain" {
  description = "Subdomain (e.g., video-compression)"
  type        = string
  default     = "video-compression"
}

variable "domain" {
  description = "Full domain name"
  type        = string
  default     = "video-compression.iffuso.com"
}

