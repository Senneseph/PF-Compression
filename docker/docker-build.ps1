# PowerShell script to build and run the PF-Compression PWA Docker container

# Function to display usage information
function Show-Usage {
  Write-Host "Usage: .\docker-build.ps1 [options]"
  Write-Host "Options:"
  Write-Host "  -Dev     Run in development mode with hot reloading"
  Write-Host "  -Prod    Run in production mode (default)"
  Write-Host "  -Build   Build the Docker image"
  Write-Host "  -Run     Run the Docker container"
  Write-Host "  -Help    Show this help message"
  exit 1
}

# Parse command line arguments
param(
  [switch]$Dev,
  [switch]$Prod,
  [switch]$Build,
  [switch]$Run,
  [switch]$Help
)

# Show help if requested
if ($Help) {
  Show-Usage
}

# Default values
$Mode = if ($Dev) { "dev" } else { "prod" }
$Action = if ($Build -and -not $Run) { "build" } elseif ($Run -and -not $Build) { "run" } else { "both" }

# Set the Docker Compose file based on the mode
if ($Mode -eq "dev") {
  $ComposeFile = "docker-compose.dev.yml"
  Write-Host "Running in development mode"
} else {
  $ComposeFile = "docker-compose.yml"
  Write-Host "Running in production mode"
}

# Execute the requested action
if ($Action -eq "build" -or $Action -eq "both") {
  Write-Host "Building Docker image..."
  docker-compose -f $ComposeFile build
}

if ($Action -eq "run" -or $Action -eq "both") {
  Write-Host "Running Docker container..."
  docker-compose -f $ComposeFile up
}
