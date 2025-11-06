# Quick HPA Validation Script
# Purpose: Validate current CPU-based HPA works before deploying SLO version

Write-Host "=== HPA Validation Test ===" -ForegroundColor Cyan
Write-Host "This test validates the current CPU-based HPA configuration"
Write-Host ""

# Step 1: Check pod health
Write-Host "[1/6] Checking inference pod health..." -ForegroundColor Yellow
$pods = kubectl get pods -l app=inference -o json | ConvertFrom-Json
$readyPods = ($pods.items | Where-Object { $_.status.phase -eq "Running" -and $_.status.conditions | Where-Object { $_.type -eq "Ready" -and $_.status -eq "True" } }).Count
Write-Host "Ready pods: $readyPods / $($pods.items.Count)" -ForegroundColor $(if ($readyPods -gt 0) { "Green" } else { "Red" })

if ($readyPods -eq 0) {
    Write-Host "ERROR: No ready pods! Checking logs..." -ForegroundColor Red
    kubectl get pods -l app=inference
    kubectl logs -l app=inference --tail=30
    exit 1
}

# Step 2: Test inference endpoint
Write-Host ""
Write-Host "[2/6] Testing inference endpoint..." -ForegroundColor Yellow
try {
    $testResult = kubectl exec deployment/inference -- python -c "import requests; r = requests.post('http://localhost:8000/predict', json={'inference_length': 1}); print(f'Status: {r.status_code}')" 2>&1
    if ($testResult -like "*Status: 200*") {
        Write-Host "✓ Inference endpoint responding successfully" -ForegroundColor Green
    } else {
        Write-Host "⚠ Inference endpoint returned: $testResult" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ Inference endpoint test failed: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Check HPA status
Write-Host ""
Write-Host "[3/6] Checking HPA configuration..." -ForegroundColor Yellow
$hpa = kubectl get hpa inference -o json 2>$null | ConvertFrom-Json
if ($hpa) {
    Write-Host "HPA Name: $($hpa.metadata.name)" -ForegroundColor Cyan
    Write-Host "Min Replicas: $($hpa.spec.minReplicas)" -ForegroundColor Cyan
    Write-Host "Max Replicas: $($hpa.spec.maxReplicas)" -ForegroundColor Cyan
    Write-Host "Current Replicas: $($hpa.status.currentReplicas)" -ForegroundColor Cyan
    Write-Host "Desired Replicas: $($hpa.status.desiredReplicas)" -ForegroundColor Cyan
    
    if ($hpa.status.currentMetrics) {
        $cpuMetric = $hpa.status.currentMetrics | Where-Object { $_.type -eq "Resource" -and $_.resource.name -eq "cpu" } | Select-Object -First 1
        if ($cpuMetric) {
            $currentCPU = $cpuMetric.resource.current.averageUtilization
            $targetCPU = $hpa.spec.metrics[0].resource.target.averageUtilization
            Write-Host "Current CPU: $currentCPU% / Target: $targetCPU%" -ForegroundColor Cyan
        }
    }
} else {
    Write-Host "⚠ No HPA found - creating default..." -ForegroundColor Yellow
    kubectl autoscale deployment inference --min=2 --max=15 --cpu-percent=70
}

# Step 4: Quick load test
Write-Host ""
Write-Host "[4/6] Running quick load test (30 users, 60s)..." -ForegroundColor Yellow
Write-Host "This will generate load to test HPA responsiveness" -ForegroundColor Gray

$loadTestStart = Get-Date
$loadTestOutput = kubectl exec deployment/locust-master -- sh -c "cd /home/locust && locust -f locustfile.py --headless --host=http://inference:8000 -u 30 -r 5 -t 60s" 2>&1

$aggregatedLine = $loadTestOutput | Select-String "Aggregated" | Select-Object -Last 1
if ($aggregatedLine) {
    Write-Host $aggregatedLine -ForegroundColor Green
} else {
    Write-Host "Load test output:" -ForegroundColor Yellow
    $loadTestOutput | Select-Object -Last 10 | ForEach-Object { Write-Host $_ }
}

# Step 5: Check if HPA scaled
Write-Host ""
Write-Host "[5/6] Checking if HPA scaled during load..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
$hpaAfter = kubectl get hpa inference -o json | ConvertFrom-Json
$replicasAfter = $hpaAfter.status.currentReplicas
$cpuAfter = ($hpaAfter.status.currentMetrics | Where-Object { $_.type -eq "Resource" } | Select-Object -First 1).resource.current.averageUtilization

Write-Host "Replicas after load: $replicasAfter (started with $($hpa.status.currentReplicas))" -ForegroundColor $(if ($replicasAfter -gt $hpa.status.currentReplicas) { "Green" } else { "Yellow" })
Write-Host "CPU after load: $cpuAfter%" -ForegroundColor Cyan

# Step 6: Test summary
Write-Host ""
Write-Host "[6/6] Test Summary" -ForegroundColor Yellow
Write-Host "==================" -ForegroundColor Yellow

$summary = @{
    "Pod Health" = if ($readyPods -gt 0) { "✓ PASS" } else { "✗ FAIL" }
    "Endpoint" = if ($testResult -like "*Status: 200*") { "✓ PASS" } else { "⚠ WARN" }
    "HPA Exists" = if ($hpa) { "✓ PASS" } else { "⚠ WARN" }
    "HPA Scaled" = if ($replicasAfter -gt $hpa.status.currentReplicas) { "✓ PASS" } else { "⚠ NEUTRAL" }
    "Load Test" = if ($aggregatedLine) { "✓ PASS" } else { "⚠ WARN" }
}

$summary.GetEnumerator() | ForEach-Object {
    $color = if ($_.Value -like "*✓*") { "Green" } elseif ($_.Value -like "*⚠*") { "Yellow" } else { "Red" }
    Write-Host "$($_.Key): $($_.Value)" -ForegroundColor $color
}

Write-Host ""
Write-Host "=== Next Steps ===" -ForegroundColor Cyan
if ($readyPods -gt 0 -and $testResult -like "*Status: 200*") {
    Write-Host "✓ System is healthy and ready for full test suite" -ForegroundColor Green
    Write-Host ""
    Write-Host "Run full HPA test matrix:" -ForegroundColor White
    Write-Host "  .\scripts\k8s_auto_hpa_tests.ps1 -UserCounts @(50,100,200) -WorkerCounts @(4) -TestDuration 120 -OutputDir 'reports\hpa_baseline'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Watch HPA scaling in real-time:" -ForegroundColor White
    Write-Host "  kubectl get hpa inference -w" -ForegroundColor Gray
} else {
    Write-Host "⚠ System has issues - resolve before running full tests" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Debug commands:" -ForegroundColor White
    Write-Host "  kubectl get pods -l app=inference" -ForegroundColor Gray
    Write-Host "  kubectl logs -l app=inference --tail=50" -ForegroundColor Gray
    Write-Host "  kubectl describe pod -l app=inference" -ForegroundColor Gray
}

Write-Host ""
