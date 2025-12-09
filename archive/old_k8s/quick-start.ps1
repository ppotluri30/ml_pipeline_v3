# Quick Start: Kubernetes Locust Load Testing
# This script deploys and runs the distributed Locust load test matrix

param(
    [switch]$Deploy,
    [switch]$Run,
    [switch]$Monitor,
    [switch]$Results,
    [switch]$Cleanup,
    [switch]$All
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Deploy-LocustInfrastructure {
    Write-Step "Deploying Locust Infrastructure"
    
    # Apply RBAC
    Write-Host "Creating ServiceAccount and RBAC..."
    kubectl apply -f .k8s/locust-driver-job.yaml 2>&1 | Select-String "serviceaccount|role|rolebinding"
    
    # Apply ConfigMap
    Write-Host "Creating ConfigMap with test script..."
    kubectl apply -f .k8s/locust-configmap.yaml
    
    # Deploy Master
    Write-Host "Deploying Locust Master..."
    kubectl apply -f .k8s/locust-master.yaml
    
    # Deploy Workers
    Write-Host "Deploying Locust Workers (4 replicas)..."
    kubectl apply -f .k8s/locust-worker.yaml
    
    # Wait for ready state
    Write-Host "Waiting for Locust master to be ready..."
    kubectl wait --for=condition=ready pod -l app=locust,role=master --timeout=120s
    
    Write-Host "Waiting for Locust workers to be ready..."
    kubectl wait --for=condition=ready pod -l app=locust,role=worker --timeout=120s
    
    Write-Success "Locust infrastructure deployed successfully"
    
    # Test master endpoint
    Write-Host "`nTesting Locust master endpoint..."
    Start-Sleep -Seconds 5
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:30089/" -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Success "Locust master is accessible at http://localhost:30089"
        }
    } catch {
        Write-Error-Custom "Could not reach Locust master. Check port-forward or NodePort configuration."
    }
}

function Build-DriverImage {
    Write-Step "Building Driver Image"
    
    if (!(Test-Path ".k8s/Dockerfile.driver")) {
        Write-Error-Custom "Dockerfile.driver not found in .k8s directory"
        return $false
    }
    
    Write-Host "Building locust-driver:latest..."
    docker build -t locust-driver:latest -f .k8s/Dockerfile.driver .k8s/
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Driver image built successfully"
        return $true
    } else {
        Write-Error-Custom "Failed to build driver image"
        return $false
    }
}

function Run-LoadTestMatrix {
    Write-Step "Running Load Test Matrix"
    
    # Build driver image if not exists
    $imageExists = docker images locust-driver:latest --format "{{.Repository}}" | Where-Object { $_ -eq "locust-driver" }
    if (!$imageExists) {
        Write-Host "Driver image not found. Building..."
        if (!(Build-DriverImage)) {
            return
        }
    }
    
    # Delete existing job
    Write-Host "Cleaning up previous job..."
    kubectl delete job locust-load-test-matrix --ignore-not-found=true 2>$null
    Start-Sleep -Seconds 3
    
    # Create new job
    Write-Host "Starting load test job..."
    kubectl apply -f .k8s/locust-driver-job-simple.yaml
    
    Write-Success "Load test job started"
    Write-Host "`nMonitor progress with: kubectl logs -f job/locust-load-test-matrix"
    Write-Host "Or run: .\quick-start.ps1 -Monitor"
}

function Monitor-LoadTest {
    Write-Step "Monitoring Load Test Progress"
    
    # Check job status
    $jobStatus = kubectl get job locust-load-test-matrix -o jsonpath='{.status.active}' 2>$null
    if (!$jobStatus) {
        Write-Error-Custom "Job not found or not running"
        Write-Host "Start the job first: .\quick-start.ps1 -Run"
        return
    }
    
    Write-Host "Following job logs (Ctrl+C to exit)...`n"
    kubectl logs -f job/locust-load-test-matrix
}

function Get-TestResults {
    Write-Step "Retrieving Test Results"
    
    # Find pod
    $pod = kubectl get pod -l job-name=locust-load-test-matrix -o jsonpath='{.items[0].metadata.name}' 2>$null
    if (!$pod) {
        Write-Error-Custom "No pod found for job. Job may not have started yet."
        return
    }
    
    # Check job completion
    $jobComplete = kubectl get job locust-load-test-matrix -o jsonpath='{.status.succeeded}' 2>$null
    if (!$jobComplete -or $jobComplete -eq "0") {
        Write-Host "⚠️  Job is still running or failed. Results may be incomplete." -ForegroundColor Yellow
        Write-Host "Check status: kubectl get job locust-load-test-matrix"
    }
    
    # Copy results
    Write-Host "Copying results from pod: $pod"
    
    New-Item -ItemType Directory -Force -Path ".\locust-results" | Out-Null
    
    kubectl cp "${pod}:/results/auto_summary.csv" ".\locust-results\auto_summary.csv" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Downloaded: .\locust-results\auto_summary.csv"
    } else {
        Write-Host "⚠️  CSV not found (test may still be running)" -ForegroundColor Yellow
    }
    
    kubectl cp "${pod}:/results/auto_summary.md" ".\locust-results\auto_summary.md" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Downloaded: .\locust-results\auto_summary.md"
        
        # Display summary
        Write-Host "`n--- Results Summary ---"
        Get-Content ".\locust-results\auto_summary.md" | Select-Object -First 30
        Write-Host "`nFull report: .\locust-results\auto_summary.md"
    } else {
        Write-Host "⚠️  Markdown not found (test may still be running)" -ForegroundColor Yellow
    }
}

function Cleanup-Resources {
    Write-Step "Cleaning Up Locust Resources"
    
    Write-Host "Deleting job..."
    kubectl delete job locust-load-test-matrix --ignore-not-found=true
    
    Write-Host "Deleting deployments..."
    kubectl delete deployment locust-master locust-worker --ignore-not-found=true
    
    Write-Host "Deleting service..."
    kubectl delete service locust-master --ignore-not-found=true
    
    Write-Host "Deleting ConfigMap..."
    kubectl delete configmap locust-scripts --ignore-not-found=true
    
    Write-Host "Deleting RBAC resources..."
    kubectl delete serviceaccount locust-driver --ignore-not-found=true
    kubectl delete role locust-driver-role --ignore-not-found=true
    kubectl delete rolebinding locust-driver-rolebinding --ignore-not-found=true
    
    Write-Success "Cleanup complete"
}

# Main execution
try {
    if ($All) {
        Deploy-LocustInfrastructure
        Start-Sleep -Seconds 10
        Run-LoadTestMatrix
        Write-Host "`n⏳ Test will take ~54 minutes for full matrix (24 scenarios)"
        Write-Host "Monitor with: .\quick-start.ps1 -Monitor"
        Write-Host "Get results: .\quick-start.ps1 -Results"
    }
    elseif ($Deploy) {
        Deploy-LocustInfrastructure
    }
    elseif ($Run) {
        Run-LoadTestMatrix
    }
    elseif ($Monitor) {
        Monitor-LoadTest
    }
    elseif ($Results) {
        Get-TestResults
    }
    elseif ($Cleanup) {
        Cleanup-Resources
    }
    else {
        Write-Host @"

Kubernetes Locust Load Testing - Quick Start

Usage:
    .\quick-start.ps1 -All        Deploy infrastructure and run full test matrix
    .\quick-start.ps1 -Deploy     Deploy Locust master and workers only
    .\quick-start.ps1 -Run        Start load test job (builds driver if needed)
    .\quick-start.ps1 -Monitor    Follow job logs in real-time
    .\quick-start.ps1 -Results    Download and display test results
    .\quick-start.ps1 -Cleanup    Remove all Locust resources

Examples:
    # Full automated run (deploy + test):
    .\quick-start.ps1 -All

    # Step-by-step:
    .\quick-start.ps1 -Deploy
    .\quick-start.ps1 -Run
    .\quick-start.ps1 -Monitor
    .\quick-start.ps1 -Results

    # Cleanup when done:
    .\quick-start.ps1 -Cleanup

Prerequisites:
    - kubectl configured with cluster access
    - Docker Desktop with Kubernetes enabled
    - Inference deployment already running

Test Matrix:
    - Replicas: [1, 2, 4, 8]
    - Workers: [4, 8]
    - Users: [200, 400, 800]
    - Total: 24 scenarios (~54 minutes)

"@
    }
}
catch {
    Write-Error-Custom "Error: $_"
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
}
