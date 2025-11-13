# Integrated Load Test with Live Monitoring
# Runs Locust load test while simultaneously monitoring all latency metrics
#
# Usage: .\Run-MonitoredLoadTest.ps1 -Users 150 -SpawnRate 10 -Duration 90

param(
    [int]$Users = 150,
    [int]$SpawnRate = 10,
    [int]$Duration = 90,
    [int]$MonitorInterval = 5
)

$ErrorActionPreference = "Continue"

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " Integrated Load Test with Live Latency Monitoring" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "Load Test Config:" -ForegroundColor Yellow
Write-Host "  Users: $Users" -ForegroundColor White
Write-Host "  Spawn Rate: $SpawnRate users/sec" -ForegroundColor White
Write-Host "  Duration: $Duration seconds" -ForegroundColor White
Write-Host "  Monitor Interval: $MonitorInterval seconds" -ForegroundColor White
Write-Host ""

# Pre-flight checks
Write-Host "Pre-flight Checks:" -ForegroundColor Yellow

# Check Locust pod
Write-Host "  Checking Locust master pod..." -NoNewline
# Use flexible pod detection with multiple fallback strategies
$locustPod = $null

# Strategy 1: Label selector with "master" in pod name
$locustPod = kubectl get pods -l app=locust -o name 2>$null | Where-Object { $_ -match 'locust-master.*-[a-z0-9]+-[a-z0-9]+$' } | Select-Object -First 1

# Strategy 2: Fallback to any pod with "locust" and "master" in name
if (-not $locustPod) {
    $locustPod = kubectl get pods -o name 2>$null | Where-Object { $_ -match 'locust.*master.*-[a-z0-9]+-[a-z0-9]+$' } | Select-Object -First 1
}

# Strategy 3: Just check for locust-master prefix (most lenient)
if (-not $locustPod) {
    $locustPod = kubectl get pods -o name 2>$null | Where-Object { $_ -match '^pod/locust-master' } | Select-Object -First 1
}

if ($locustPod) {
    $podName = ($locustPod -replace 'pod/', '')
    Write-Host " OK ($podName)" -ForegroundColor Green
} else {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "ERROR: Locust master pod not found!" -ForegroundColor Red
    Write-Host "Available Locust pods:" -ForegroundColor Yellow
    kubectl get pods -l app=locust
    exit 1
}

# Check inference pods
Write-Host "  Checking inference pods..." -NoNewline
$inferencePods = kubectl get pods -l app=inference --field-selector=status.phase=Running -o jsonpath='{.items[*].metadata.name}' 2>$null
$inferenceCount = ($inferencePods -split ' ').Count
if ($inferenceCount -gt 0) {
    Write-Host " OK ($inferenceCount running)" -ForegroundColor Green
} else {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "ERROR: No running inference pods found!" -ForegroundColor Red
    exit 1
}

# Check Prometheus
Write-Host "  Checking Prometheus..." -NoNewline
$promPod = kubectl get pods -l app.kubernetes.io/name=prometheus -o jsonpath='{.items[0].metadata.name}' 2>$null
if ($promPod) {
    Write-Host " OK ($promPod)" -ForegroundColor Green
} else {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "ERROR: Prometheus pod not found!" -ForegroundColor Red
    exit 1
}

# Check KEDA
Write-Host "  Checking KEDA..." -NoNewline
$kedaOp = kubectl get pods -n keda -l app=keda-operator -o jsonpath='{.items[0].metadata.name}' 2>$null
if ($kedaOp) {
    Write-Host " OK ($kedaOp)" -ForegroundColor Green
} else {
    Write-Host " WARNING" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting load test + monitoring in 3 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Start monitoring job in background
$monitoringScript = {
    param($duration, $interval, $outputFile, $scriptPath)
    
    Set-Location -Path (Split-Path $scriptPath)
    & $scriptPath -Duration $duration -SampleInterval $interval -OutputCsv $outputFile
}

$scriptPath = Join-Path $PSScriptRoot "Monitor-LiveLatency.ps1"
$monitoringCsv = "monitoring-$(Get-Date -Format 'yyyyMMdd-HHmmss').csv"
$monitoringJob = Start-Job -ScriptBlock $monitoringScript -ArgumentList $Duration, $MonitorInterval, $monitoringCsv, $scriptPath

Write-Host "Monitoring job started (Job ID: $($monitoringJob.Id))" -ForegroundColor Green
Write-Host "Output will be saved to: $monitoringCsv" -ForegroundColor White
Write-Host ""

# Wait 5 seconds for monitoring to initialize
Start-Sleep -Seconds 5

# Start load test in foreground
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " Starting Locust Load Test" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

$loadTestCommand = "kubectl exec deployment/locust-master -- locust --headless --users $Users --spawn-rate $SpawnRate --run-time $($Duration)s --host http://inference:8000"

Write-Host "Executing: $loadTestCommand" -ForegroundColor Gray
Write-Host ""

Invoke-Expression $loadTestCommand

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " Load Test Complete" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Wait for monitoring job to finish
Write-Host "Waiting for monitoring job to complete..." -ForegroundColor Yellow
Wait-Job -Job $monitoringJob -Timeout 30 | Out-Null

# Get monitoring job output
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " Monitoring Results" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

Receive-Job -Job $monitoringJob

# Clean up job
Remove-Job -Job $monitoringJob -Force

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " Test Complete!" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitoring data saved to: $monitoringCsv" -ForegroundColor Green
Write-Host ""
Write-Host "To view the CSV:" -ForegroundColor Yellow
Write-Host "  Import-Csv $monitoringCsv | Format-Table" -ForegroundColor White
Write-Host ""
