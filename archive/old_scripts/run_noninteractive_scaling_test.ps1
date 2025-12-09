#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Non-interactive KEDA scaling test with parallel Locust load and passive telemetry collection.

.DESCRIPTION
    Runs Locust load test asynchronously while collecting scaling metrics in parallel.
    No interference with running test - all metrics collected passively via kubectl/Prometheus.

.PARAMETER Users
    Number of concurrent users for Locust test (default: 150)

.PARAMETER SpawnRate
    User spawn rate per second (default: 10)

.PARAMETER Duration
    Test duration in seconds (default: 300 = 5 minutes)

.PARAMETER SampleInterval
    Telemetry collection interval in seconds (default: 10)

.PARAMETER OutputDir
    Directory for output files (default: ./scaling_test_results)

.EXAMPLE
    .\run_noninteractive_scaling_test.ps1 -Users 150 -Duration 300
#>

param(
    [int]$Users = 150,
    [int]$SpawnRate = 10,
    [int]$Duration = 300,
    [int]$SampleInterval = 10,
    [string]$OutputDir = "./scaling_test_results"
)

$ErrorActionPreference = "Continue"

# Create output directory
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$testDir = Join-Path $OutputDir "test_${timestamp}"
New-Item -ItemType Directory -Path $testDir -Force | Out-Null

$telemetryFile = Join-Path $testDir "telemetry.csv"
$locustLogFile = Join-Path $testDir "locust_output.log"
$summaryFile = Join-Path $testDir "test_summary.txt"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "KEDA Non-Interactive Scaling Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Users: $Users | Spawn Rate: $SpawnRate/s | Duration: ${Duration}s"
Write-Host "Sample Interval: ${SampleInterval}s"
Write-Host "Output Directory: $testDir"
Write-Host ""

# Initialize telemetry CSV
"Timestamp,Elapsed_Sec,Replicas,CPU_Avg_Pct,Mem_Avg_Pct,p95_Latency_ms,Queue_Len_Avg,HPA_Status" | Out-File -FilePath $telemetryFile -Encoding UTF8

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Pre-test baseline check..." -ForegroundColor Yellow

# Get initial state
$initialReplicas = (kubectl get pods -l app=inference --no-headers 2>$null | Measure-Object).Count
$initialHPA = kubectl get hpa keda-hpa-inference-slo-scaler -o jsonpath='{.status.currentReplicas}' 2>$null

Write-Host "  Initial replicas: $initialReplicas"
Write-Host "  HPA current replicas: $initialHPA"
Write-Host ""

# Start Locust test asynchronously
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting Locust load test (background)..." -ForegroundColor Green

$locustCommand = "kubectl exec -n default deployment/locust-master -- locust --headless --users $Users --spawn-rate $SpawnRate --run-time ${Duration}s --host http://inference:8000 --only-summary"

# Start Locust in background job
$locustJob = Start-Job -ScriptBlock {
    param($cmd, $logFile)
    $output = Invoke-Expression $cmd 2>&1
    $output | Out-File -FilePath $logFile -Encoding UTF8
    return $output
} -ArgumentList $locustCommand, $locustLogFile

Write-Host "  Locust job started: ID=$($locustJob.Id)" -ForegroundColor Green
Write-Host "  Waiting 15s for test ramp-up..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Telemetry collection loop
Write-Host ""
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting telemetry collection..." -ForegroundColor Cyan
Write-Host "  Collecting metrics every ${SampleInterval}s until test completes..."
Write-Host "  (Press Ctrl+C to abort early - test will continue)"
Write-Host ""

$testStartTime = Get-Date
$sampleCount = 0

while ($locustJob.State -eq 'Running') {
    $elapsed = [math]::Round(((Get-Date) - $testStartTime).TotalSeconds, 0)
    
    # Collect metrics (all non-blocking queries)
    try {
        # Pod count
        $replicas = (kubectl get pods -l app=inference --no-headers 2>$null | Measure-Object).Count
        if (-not $replicas) { $replicas = 0 }
        
        # CPU and Memory from metrics-server
        $metricsRaw = kubectl top pods -l app=inference --no-headers 2>$null
        $cpuValues = @()
        $memValues = @()
        
        if ($metricsRaw) {
            foreach ($line in $metricsRaw) {
                $parts = $line -split '\s+'
                if ($parts.Length -ge 3) {
                    # CPU in millicores (e.g., "245m")
                    $cpuStr = $parts[1] -replace 'm', ''
                    if ($cpuStr -match '^\d+$') {
                        $cpuValues += [int]$cpuStr
                    }
                    # Memory in Mi (e.g., "1234Mi")
                    $memStr = $parts[2] -replace 'Mi', ''
                    if ($memStr -match '^\d+$') {
                        $memValues += [int]$memStr
                    }
                }
            }
        }
        
        $cpuAvg = if ($cpuValues.Count -gt 0) { 
            [math]::Round(($cpuValues | Measure-Object -Average).Average, 0) 
        } else { 0 }
        
        $memAvg = if ($memValues.Count -gt 0) { 
            [math]::Round(($memValues | Measure-Object -Average).Average, 0) 
        } else { 0 }
        
        # Prometheus p95 latency
        $p95Query = "histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[1m])) by (le))"
        $p95Raw = kubectl exec -n default prometheus-server-c568bf4db-zmk2t -c prometheus-server -- `
            wget -qO- "http://localhost:9090/api/v1/query?query=$([uri]::EscapeDataString($p95Query))" 2>$null
        
        $p95Latency = 0
        if ($p95Raw) {
            try {
                $p95Json = $p95Raw | ConvertFrom-Json
                $p95Value = $p95Json.data.result[0].value[1]
                if ($p95Value) {
                    $p95Latency = [math]::Round([double]$p95Value * 1000, 0)  # Convert to ms
                }
            } catch {
                $p95Latency = 0
            }
        }
        
        # Prometheus queue length
        $queueQuery = "avg(inference_queue_len)"
        $queueRaw = kubectl exec -n default prometheus-server-c568bf4db-zmk2t -c prometheus-server -- `
            wget -qO- "http://localhost:9090/api/v1/query?query=$([uri]::EscapeDataString($queueQuery))" 2>$null
        
        $queueLen = 0
        if ($queueRaw) {
            try {
                $queueJson = $queueRaw | ConvertFrom-Json
                $queueValue = $queueJson.data.result[0].value[1]
                if ($queueValue) {
                    $queueLen = [math]::Round([double]$queueValue, 1)
                }
            } catch {
                $queueLen = 0
            }
        }
        
        # HPA status
        $hpaStatus = kubectl get hpa keda-hpa-inference-slo-scaler -o jsonpath='{.status.conditions[?(@.type=="ScalingActive")].status}' 2>$null
        if (-not $hpaStatus) { $hpaStatus = "Unknown" }
        
        # Write to CSV
        $timestamp = Get-Date -Format "HH:mm:ss"
        "$timestamp,$elapsed,$replicas,$cpuAvg,$memAvg,$p95Latency,$queueLen,$hpaStatus" | Out-File -FilePath $telemetryFile -Append -Encoding UTF8
        
        $sampleCount++
        
        # Progress indicator (no scrolling)
        Write-Host "`r  Sample $sampleCount | ${elapsed}s | Pods: $replicas | CPU: ${cpuAvg}m | p95: ${p95Latency}ms | Queue: $queueLen      " -NoNewline -ForegroundColor Gray
        
    } catch {
        Write-Host "`r  Sample failed: $_      " -NoNewline -ForegroundColor Red
    }
    
    # Wait for next sample
    Start-Sleep -Seconds $SampleInterval
}

Write-Host ""
Write-Host ""

# Wait for Locust job to complete
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Waiting for Locust test to finish..." -ForegroundColor Yellow
$locustResult = Receive-Job -Job $locustJob -Wait

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Test complete! Processing results..." -ForegroundColor Green
Write-Host ""

# Parse Locust output
$locustStats = Get-Content $locustLogFile -Raw

# Extract summary stats
$aggregatedLine = $locustStats | Select-String -Pattern "Aggregated\s+\d+" | Select-Object -Last 1
$statsMatch = $aggregatedLine -match "Aggregated\s+(\d+)\s+(\d+)\([\d.]+%\)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)"

if ($statsMatch) {
    $totalRequests = $Matches[1]
    $failures = $Matches[2]
    $avgLatency = $Matches[3]
    $minLatency = $Matches[4]
    $maxLatency = $Matches[5]
    $medianLatency = $Matches[6]
    $rps = $Matches[7]
    
    $successRate = if ($totalRequests -gt 0) { 
        [math]::Round((1 - ([int]$failures / [int]$totalRequests)) * 100, 2) 
    } else { 0 }
} else {
    $totalRequests = "N/A"
    $failures = "N/A"
    $avgLatency = "N/A"
    $medianLatency = "N/A"
    $rps = "N/A"
    $successRate = "N/A"
}

# Load telemetry data
$telemetryData = Import-Csv $telemetryFile

# Calculate scaling statistics
$replicaCounts = $telemetryData | Select-Object -ExpandProperty Replicas | ForEach-Object { [int]$_ }
$minReplicas = ($replicaCounts | Measure-Object -Minimum).Minimum
$maxReplicas = ($replicaCounts | Measure-Object -Maximum).Maximum
$avgReplicas = [math]::Round(($replicaCounts | Measure-Object -Average).Average, 1)

$p95Values = $telemetryData | Select-Object -ExpandProperty p95_Latency_ms | ForEach-Object { 
    if ($_ -match '^\d+$') { [int]$_ } else { 0 }
}
$maxP95 = ($p95Values | Measure-Object -Maximum).Maximum
$avgP95 = [math]::Round(($p95Values | Where-Object { $_ -gt 0 } | Measure-Object -Average).Average, 0)

$cpuValues = $telemetryData | Select-Object -ExpandProperty CPU_Avg_Pct | ForEach-Object { [int]$_ }
$maxCPU = ($cpuValues | Measure-Object -Maximum).Maximum
$avgCPU = [math]::Round(($cpuValues | Measure-Object -Average).Average, 0)

# Find first scale-up event
$scaleUpTime = $null
$scaleDownTime = $null
for ($i = 1; $i -lt $telemetryData.Count; $i++) {
    $prev = [int]$telemetryData[$i-1].Replicas
    $curr = [int]$telemetryData[$i].Replicas
    
    if ($curr -gt $prev -and -not $scaleUpTime) {
        $scaleUpTime = [int]$telemetryData[$i].Elapsed_Sec
    }
    if ($curr -lt $prev -and -not $scaleDownTime) {
        $scaleDownTime = [int]$telemetryData[$i].Elapsed_Sec
    }
}

# Generate summary report
$summary = @"
========================================
KEDA SCALING TEST SUMMARY
========================================
Test Configuration:
  Users: $Users
  Spawn Rate: $SpawnRate/s
  Duration: ${Duration}s
  Sample Interval: ${SampleInterval}s
  
Test Execution:
  Start Time: $($testStartTime.ToString('yyyy-MM-dd HH:mm:ss'))
  End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
  Samples Collected: $sampleCount

----------------------------------------
LOCUST LOAD TEST RESULTS
----------------------------------------
  Total Requests: $totalRequests
  Failures: $failures
  Success Rate: $successRate%
  Requests/sec: $rps
  
  Latency:
    Average: ${avgLatency}ms
    Median: ${medianLatency}ms
    Min: ${minLatency}ms
    Max: ${maxLatency}ms

----------------------------------------
SCALING BEHAVIOR
----------------------------------------
  Replica Count:
    Initial: $initialReplicas
    Minimum: $minReplicas
    Maximum: $maxReplicas
    Average: $avgReplicas
    Final: $($telemetryData[-1].Replicas)
  
  Time to First Scale-Up: $(if ($scaleUpTime) { "${scaleUpTime}s" } else { "No scale-up" })
  Time to First Scale-Down: $(if ($scaleDownTime) { "${scaleDownTime}s" } else { "No scale-down" })

----------------------------------------
PERFORMANCE METRICS
----------------------------------------
  CPU Utilization (avg across pods):
    Average: ${avgCPU}m
    Maximum: ${maxCPU}m
  
  p95 Latency:
    Average: ${avgP95}ms
    Maximum: ${maxP95}ms

----------------------------------------
SCALING TRIGGER ANALYSIS
----------------------------------------
"@

# Analyze if triggers fired
$latencyTriggered = $maxP95 -gt 500
$cpuTriggered = $maxCPU -gt 850  # 85% of 1000m

$summary += @"
  Latency Trigger (>500ms): $(if ($latencyTriggered) { "YES (max: ${maxP95}ms)" } else { "NO (max: ${maxP95}ms)" })
  CPU Trigger (>85%): $(if ($cpuTriggered) { "YES (max: ${maxCPU}m)" } else { "NO (max: ${maxCPU}m)" })
  
  Scaling Responsiveness:
$(if ($scaleUpTime) {
"    - Scale-up initiated at ${scaleUpTime}s"
"    - Replicas increased: $minReplicas → $maxReplicas"
} else {
"    - No scale-up events detected"
"    - Workload handled by $minReplicas pods"
})

----------------------------------------
RECOMMENDATIONS
----------------------------------------
"@

# Generate recommendations
if (-not $scaleUpTime -and $maxP95 -lt 500) {
    $summary += "  [OK] System stable - no scaling needed`n"
    $summary += "  [OK] Latency well below threshold (${maxP95}ms < 500ms)`n"
    $summary += "  [*] Consider increasing load or lowering latency threshold to test scaling`n"
}

if ($scaleUpTime -and $scaleUpTime -gt 60) {
    $summary += "  [!] Slow scale-up (${scaleUpTime}s)`n"
    $summary += "  [*] Consider reducing stabilizationWindowSeconds (currently 60s)`n"
    $summary += "  [*] Or lowering activation thresholds for earlier intervention`n"
}

if ($maxP95 -gt 1000) {
    $summary += "  [!] High p95 latency detected (${maxP95}ms)`n"
    $summary += "  [*] Lower latency threshold from 500ms to 400ms`n"
    $summary += "  [*] Increase scaleUp policy from 2 pods to 3 pods per cycle`n"
}

if ($maxCPU -lt 500) {
    $summary += "  [*] CPU utilization low (${maxCPU}m) - pods under-utilized`n"
    $summary += "  [*] Consider reducing resource requests for better efficiency`n"
}

$summary += @"

----------------------------------------
OUTPUT FILES
----------------------------------------
  Telemetry Data: $telemetryFile
  Locust Log: $locustLogFile
  Summary: $summaryFile

========================================
"@

# Write summary to file and console
$summary | Out-File -FilePath $summaryFile -Encoding UTF8
Write-Host $summary -ForegroundColor Cyan

# Display telemetry table (first 10 and last 10 rows)
Write-Host ""
Write-Host "TELEMETRY TIMELINE (First 10 samples):" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
$telemetryData | Select-Object -First 10 | Format-Table -AutoSize

if ($telemetryData.Count -gt 20) {
    Write-Host ""
    Write-Host "TELEMETRY TIMELINE (Last 10 samples):" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    $telemetryData | Select-Object -Last 10 | Format-Table -AutoSize
}

Write-Host ""
Write-Host "[DONE] Test complete! Results saved to: $testDir" -ForegroundColor Green
Write-Host ""

# Cleanup
Remove-Job -Job $locustJob -Force

return @{
    TestDir = $testDir
    TotalRequests = $totalRequests
    SuccessRate = $successRate
    RPS = $rps
    MaxReplicas = $maxReplicas
    MaxP95Latency = $maxP95
    ScaleUpTime = $scaleUpTime
}
