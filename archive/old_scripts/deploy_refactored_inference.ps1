# Deploy refactored inference architecture to Kubernetes
# Run from project root: .\deploy_refactored_inference.ps1

param(
    [switch]$SkipBuild = $false,
    [switch]$DeleteOld = $true
)

Write-Host "Deploying refactored inference architecture..." -ForegroundColor Cyan

# Step 1: Build images (unless skipped)
if (-not $SkipBuild) {
    Write-Host "`n=== Building Docker images ===" -ForegroundColor Yellow
    .\build_inference_images.ps1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed, aborting deployment" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Delete old inference deployment and HPA (if requested)
if ($DeleteOld) {
    Write-Host "`n=== Removing old inference deployment ===" -ForegroundColor Yellow
    
    Write-Host "Deleting old HPA: inference-hpa"
    kubectl delete hpa inference-hpa --ignore-not-found=true
    
    Write-Host "Deleting old deployment: inference"
    kubectl delete deployment inference --ignore-not-found=true
    
    Write-Host "Deleting old service: inference (if exists)"
    kubectl delete service inference --ignore-not-found=true
    
    Write-Host "Waiting for pods to terminate..."
    Start-Sleep -Seconds 10
}

# Step 3: Deploy new architecture
Write-Host "`n=== Deploying new components ===" -ForegroundColor Yellow

Write-Host "`n1. Deploying inference-worker (Kafka consumer)..."
kubectl apply -f .k8s\inference-worker-deployment.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to deploy inference-worker" -ForegroundColor Red
    exit 1
}

Write-Host "`n2. Deploying inference-http (HTTP server)..."
kubectl apply -f .k8s\inference-http-deployment.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to deploy inference-http" -ForegroundColor Red
    exit 1
}

Write-Host "`n3. Deploying inference-http HPA..."
kubectl apply -f .k8s\inference-http-hpa.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to deploy inference-http-hpa" -ForegroundColor Red
    exit 1
}

# Step 4: Wait for rollout
Write-Host "`n=== Waiting for deployments ===" -ForegroundColor Yellow

Write-Host "Waiting for inference-worker rollout..."
kubectl rollout status deployment/inference-worker --timeout=120s

Write-Host "Waiting for inference-http rollout..."
kubectl rollout status deployment/inference-http --timeout=120s

# Step 5: Verify deployment
Write-Host "`n=== Deployment Status ===" -ForegroundColor Cyan

kubectl get deployments -l 'component in (http-server,kafka-consumer)'
Write-Host ""
kubectl get pods -l 'component in (http-server,kafka-consumer)'
Write-Host ""
kubectl get hpa inference-http-hpa
Write-Host ""
kubectl get service inference-http

Write-Host "`n--- Refactored inference architecture deployed successfully!" -ForegroundColor Green
Write-Host "`nNext steps:"
Write-Host "  1. Run validation: .\validate_refactored_inference.ps1"
Write-Host "  2. Monitor CPU usage: kubectl top pods -l app=inference-http"
Write-Host "  3. Check HPA scaling: kubectl get hpa inference-http-hpa"
