# Hybrid Autoscaling Load Test
$locustPod = "locust-master-7589855596-tfwg9"
$testDuration = 90
$users = 100
$spawnRate = 10

Write-Host "=== Starting Hybrid Autoscaling Validation Test ===" -ForegroundColor Cyan
Write-Host "Configuration: $users users, $spawnRate/s spawn rate, ${testDuration}s duration" -ForegroundColor Yellow
Write-Host ""

# Start load test
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting Locust swarm..." -ForegroundColor Green

# Start port-forward in background
$portForwardJob = Start-Job -ScriptBlock {
    kubectl port-forward svc/locust-master 8089:8089
}
Start-Sleep -Seconds 5

$body = @{
    user_count = $users
    spawn_rate = $spawnRate
    host = "http://inference:8000"
}

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8089/swarm" -Method POST -Body $body -ContentType "application/x-www-form-urlencoded" -UseBasicParsing
    if ($response.StatusCode -ne 200) {
        Write-Host "Failed to start Locust swarm: $($response.StatusCode)" -ForegroundColor Red
        Stop-Job $portForwardJob; Remove-Job $portForwardJob
        exit 1
    }
} catch {
    Write-Host "Failed to start Locust swarm: $_" -ForegroundColor Red
    Stop-Job $portForwardJob; Remove-Job $portForwardJob
    exit 1
}

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Load test started. Monitoring for ${testDuration}s..." -ForegroundColor Green
Write-Host ""

# Create results file
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsFile = ".\autoscaling_telemetry_$timestamp.csv"
"Timestamp,Replicas,AvgCPU%,P95Latency,RequestRate" | Out-File -FilePath $resultsFile -Encoding utf8

Write-Host "Time     | Replicas | Avg CPU% | P95 Latency | Req/s | Status" -ForegroundColor Cyan
Write-Host "---------|----------|----------|-------------|-------|--------" -ForegroundColor Cyan

$startTime = Get-Date
$endTime = $startTime.AddSeconds($testDuration + 10)

while ((Get-Date) -lt $endTime) {
    $elapsed = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 0)
    
    # Get replica count
    $replicas = (kubectl get pods -l app=inference --no-headers 2>$null | Measure-Object).Count
    
    # Get CPU metrics
    $cpuMetrics = kubectl top pods -l app=inference --no-headers 2>$null | ForEach-Object {
        if ($_ -match '(\d+)m\s') {
            [int]$matches[1]
        }
    }
    $avgCpu = if ($cpuMetrics) { [math]::Round(($cpuMetrics | Measure-Object -Average).Average / 10, 1) } else { 0 }
    
    # Get Locust stats
    try {
        $stats = Invoke-RestMethod -Uri "http://localhost:8089/stats/requests" -Method GET -TimeoutSec 2 -ErrorAction Stop
        $p95 = if ($stats.stats[0]."95%ile") { [math]::Round($stats.stats[0]."95%ile" / 1000, 2) } else { 0 }
        $rps = if ($stats.stats[0].current_rps) { [math]::Round($stats.stats[0].current_rps, 1) } else { 0 }
        $status = if ($stats.state -eq "running") { "RUN" } else { $stats.state.ToUpper() }
    } catch {
        $p95 = 0
        $rps = 0
        $status = "N/A"
    }
    
    # Display and log
    $timeStr = "{0:D3}s" -f $elapsed
    Write-Host ("{0,-8} | {1,8} | {2,8}% | {3,10}s | {4,5} | {5}" -f $timeStr, $replicas, $avgCpu, $p95, $rps, $status)
    
    "$((Get-Date).ToString('HH:mm:ss')),$replicas,$avgCpu,$p95,$rps" | Out-File -FilePath $resultsFile -Append -Encoding utf8
    
    Start-Sleep -Seconds 5
}

Write-Host ""
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Stopping load test..." -ForegroundColor Green
try {
    Invoke-WebRequest -Uri "http://localhost:8089/stop" -Method GET -UseBasicParsing -TimeoutSec 5 | Out-Null
} catch {
    Write-Host "Warning: Could not stop Locust cleanly" -ForegroundColor Yellow
}

# Stop port-forward job
Stop-Job $portForwardJob 2>$null
Remove-Job $portForwardJob 2>$null

Write-Host ""
Write-Host "=== Test Complete ===" -ForegroundColor Cyan
Write-Host "Results saved to: $resultsFile" -ForegroundColor Yellow

# Final summary
$finalReplicas = (kubectl get pods -l app=inference --no-headers | Measure-Object).Count
$hpaStatus = kubectl get hpa inference-hpa --no-headers
Write-Host ""
Write-Host "Final State:" -ForegroundColor Cyan
Write-Host "  Replicas: $finalReplicas" -ForegroundColor White
Write-Host "  HPA: $hpaStatus" -ForegroundColor White
