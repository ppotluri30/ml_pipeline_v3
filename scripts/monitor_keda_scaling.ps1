# KEDA Load Test Monitoring Script
# Captures real-time scaling metrics during Locust load test

param(
    [int]$DurationSeconds = 180,
    [int]$SampleInterval = 5
)

$resultsFile = "keda_scaling_timeline.csv"
$prometheusUrl = "http://localhost:9091"

Write-Host "Starting KEDA scaling monitor for $DurationSeconds seconds..." -ForegroundColor Cyan
Write-Host "Results will be saved to: $resultsFile" -ForegroundColor Yellow

# Create CSV header
"Timestamp,Replicas,p95_Latency_ms,Avg_Queue_Len,CPU_Percent,Active_Triggers" | Out-File -FilePath $resultsFile -Encoding utf8

$startTime = Get-Date
$endTime = $startTime.AddSeconds($DurationSeconds)

while ((Get-Date) -lt $endTime) {
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    # Get replica count
    try {
        $pods = kubectl get pods -l app=inference -o json | ConvertFrom-Json
        $replicas = ($pods.items | Where-Object { $_.status.phase -eq "Running" }).Count
    } catch {
        $replicas = "ERROR"
    }
    
    # Get p95 latency from Prometheus
    try {
        $latencyQuery = "histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[1m])) by (le))"
        $latencyUrl = "$prometheusUrl/api/v1/query?query=$([System.Uri]::EscapeDataString($latencyQuery))"
        $latencyResult = Invoke-RestMethod -Uri $latencyUrl -Method GET -ErrorAction Stop
        $p95Latency = [math]::Round([double]$latencyResult.data.result[0].value[1] * 1000, 2)
    } catch {
        $p95Latency = "N/A"
    }
    
    # Get average queue length
    try {
        $queueQuery = "avg(inference_queue_len)"
        $queueUrl = "$prometheusUrl/api/v1/query?query=$([System.Uri]::EscapeDataString($queueQuery))"
        $queueResult = Invoke-RestMethod -Uri $queueUrl -Method GET -ErrorAction Stop
        $avgQueue = [math]::Round([double]$queueResult.data.result[0].value[1], 1)
    } catch {
        $avgQueue = "N/A"
    }
    
    # Get CPU utilization
    try {
        $cpuOutput = kubectl top pods -l app=inference --no-headers
        $cpuValues = $cpuOutput -split "`n" | ForEach-Object {
            if ($_ -match '(\d+)m') { [int]$matches[1] }
        }
        $avgCpu = if ($cpuValues.Count -gt 0) { [math]::Round(($cpuValues | Measure-Object -Average).Average / 10, 1) } else { "N/A" }
    } catch {
        $avgCpu = "N/A"
    }
    
    # Get KEDA active status
    try {
        $scaledobject = kubectl get scaledobject inference-slo-scaler -o json | ConvertFrom-Json
        $activeCondition = $scaledobject.status.conditions | Where-Object { $_.type -eq "Active" }
        $activeTriggers = if ($activeCondition.status -eq "True") { "YES" } else { "NO" }
    } catch {
        $activeTriggers = "ERROR"
    }
    
    # Write to CSV and console
    $row = "$timestamp,$replicas,$p95Latency,$avgQueue,$avgCpu,$activeTriggers"
    $row | Out-File -FilePath $resultsFile -Append -Encoding utf8
    
    Write-Host "$timestamp | Replicas: $replicas | p95: ${p95Latency}ms | Queue: $avgQueue | CPU: ${avgCpu}% | Active: $activeTriggers" -ForegroundColor Green
    
    Start-Sleep -Seconds $SampleInterval
}

Write-Host "`nMonitoring complete! Results saved to: $resultsFile" -ForegroundColor Cyan
