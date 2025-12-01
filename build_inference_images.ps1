# Build script for inference HTTP and worker images
# Run from project root: .\build_inference_images.ps1

Write-Host "Building inference HTTP and worker images..." -ForegroundColor Cyan

# Build HTTP-only inference image
Write-Host "`nBuilding inference-http:latest..." -ForegroundColor Yellow
docker build -f inference_container/Dockerfile.http -t inference-http:latest .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build inference-http image" -ForegroundColor Red
    exit 1
}
Write-Host "✓ inference-http:latest built successfully" -ForegroundColor Green

# Build Kafka worker image
Write-Host "`nBuilding inference-worker:latest..." -ForegroundColor Yellow
docker build -f inference_container/Dockerfile.worker -t inference-worker:latest .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build inference-worker image" -ForegroundColor Red
    exit 1
}
Write-Host "✓ inference-worker:latest built successfully" -ForegroundColor Green

Write-Host "`nBoth images built successfully!" -ForegroundColor Cyan
Write-Host "  - inference-http:latest (HTTP-only FastAPI server)"
Write-Host "  - inference-worker:latest (Kafka consumer worker)"
