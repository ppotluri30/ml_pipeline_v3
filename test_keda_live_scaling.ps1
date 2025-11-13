# KEDA Live Scaling Test - Monitor Prometheus-Driven Scaling
# Runs Locust load test and captures real-time scaling telemetry

param(
    [int]$Users = 150,
    [int]$SpawnRate = 10,
    [int]$Duration = 180,
    [int]$SampleInterval = 10
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsFile = "keda_live_scaling_$timestamp.csv"

Write-Host "" -ForegroundColor Cyan
Write-Host "=== KEDA Live Scaling Test ===" -ForegroundColor Cyan
Write-Host "Users: $Users | Spawn Rate: $SpawnRate/s | Duration: ${Duration}s" -ForegroundColor Yellow
Write-Host "Results: $resultsFile" -ForegroundColor Green
Write-Host "" -ForegroundColor Cyan

# Create CSV header
"Time,Replicas,p95_Latency_ms,Avg_Queue,CPU_Avg_%,Triggers_Active" | Out-File -FilePath $resultsFile -Encoding utf8

# Start load test
Write-Host "Starting Locust load test..." -ForegroundColor Cyan
$locustJob = Start-Job -ScriptBlock {
    param($u, $sr, $dur)
    kubectl exec -n default deployment/locust-master -- locust --headless --users $u --spawn-rate $sr --run-time "${dur}s" --host http://inference:8000
} -ArgumentList $Users, $SpawnRate, $Duration

Start-Sleep -Seconds 5
Write-Host "Load test started" -ForegroundColor Green
Write-Host "" -ForegroundColor Cyan

# Monitor scaling
$endTime = (Get-Date).AddSeconds($Duration + 60)  # Extra time for cooldown observation

while ((Get-Date) -lt $endTime) {
    $now = Get-Date -Format "HH:mm:ss"
    
    # Get replica count
    try {
        $replicas = (kubectl get pods -l app=inference -o json | ConvertFrom-Json).items | 
            Where-Object { $_.status.phase -eq "Running" } | Measure-Object | Select-Object -ExpandProperty Count
    } catch { $replicas = "ERR" }
    
    # Get latency from Prometheus
    try {
        $latencyQuery = 'histogram_quantile(0.95,sum(rate(inference_latency_seconds_bucket[1m]))by(le))'
        $latencyJson = kubectl exec -n default prometheus-server-c568bf4db-zmk2t -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$latencyQuery"
        $latencyData = $latencyJson | ConvertFrom-Json
        $p95Latency = [math]::Round([double]$latencyData.data.result[0].value[1] * 1000, 0)
    } catch { $p95Latency = "N/A" }
    
    # Get queue length
    try {
        $queueQuery = 'avg(inference_queue_len)'
        $queueJson = kubectl exec -n default prometheus-server-c568bf4db-zmk2t -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$queueQuery"
        $queueData = $queueJson | ConvertFrom-Json
        $avgQueue = [math]::Round([double]$queueData.data.result[0].value[1], 1)
    } catch { $avgQueue = "N/A" }
    
    # Get CPU
    try {
        $cpuData = kubectl top pods -l app=inference --no-headers
        $cpuValues = $cpuData -split "`n" | ForEach-Object {
            if ($_ -match '(\d+)m') { [int]$matches[1] }
        }
        $avgCpu = if ($cpuValues.Count -gt 0) { [math]::Round(($cpuValues | Measure-Object -Average).Average / 10, 0) } else { "N/A" }
    } catch { $avgCpu = "N/A" }
    
    # Check KEDA active
    try {
        $soJson = kubectl get scaledobject inference-slo-scaler -o json | ConvertFrom-Json
        $activeCond = $soJson.status.conditions | Where-Object { $_.type -eq "Active" }
        $isActive = if ($activeCond.status -eq "True") { "YES" } else { "NO" }
    } catch { $isActive = "ERR" }
    
    # Log and display
    $row = "$now,$replicas,$p95Latency,$avgQueue,$avgCpu,$isActive"
    $row | Out-File -FilePath $resultsFile -Append -Encoding utf8
    
    $color = if ($isActive -eq "YES") { "Green" } else { "Yellow" }
    Write-Host "$now | Pods: $replicas | p95: ${p95Latency}ms | Queue: $avgQueue | CPU: ${avgCpu}% | Active: $isActive" -ForegroundColor $color
    
    Start-Sleep -Seconds $SampleInterval
}

# Wait for job completion
Write-Host "" -ForegroundColor Cyan
Write-Host "Waiting for load test completion..." -ForegroundColor Cyan
Wait-Job $locustJob -Timeout 30 | Out-Null
Remove-Job $locustJob -Force

Write-Host "" -ForegroundColor Cyan
Write-Host "=== Test Complete ===" -ForegroundColor Cyan
Write-Host "Results saved to: $resultsFile" -ForegroundColor Green
Write-Host "" -ForegroundColor Yellow
Write-Host "Timeline summary:" -ForegroundColor Yellow
Get-Content $resultsFile | Select-Object -First 1
Get-Content $resultsFile | Select-Object -Skip 1 | ForEach-Object {
    $fields = $_ -split ','
    if ($fields[5] -eq 'YES') {
        Write-Host $_ -ForegroundColor Green
    } else {
        Write-Host $_
    }
}
