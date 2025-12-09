#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Real-time hybrid autoscaling validation with live telemetry collection.

.DESCRIPTION
    Runs a sustained load test while collecting metrics every 5 seconds:
    - Replica count
    - CPU and Memory usage
    - p95 latency from Prometheus
    - Queue length
    
    Validates both KEDA (latency/queue) and HPA (CPU/memory) autoscalers.

.PARAMETER Users
    Number of concurrent Locust users (default: 150)

.PARAMETER SpawnRate
    User spawn rate per second (default: 10)

.PARAMETER Duration
    Test duration in seconds (default: 180)

.PARAMETER SampleInterval
    Metrics sampling interval in seconds (default: 5)

.EXAMPLE
    .\test_hybrid_autoscaling.ps1 -Users 150 -SpawnRate 10 -Duration 180 -SampleInterval 5
#>

param(
    [int]$Users = 150,
    [int]$SpawnRate = 10,
    [int]$Duration = 180,
    [int]$SampleInterval = 5
)

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = "autoscaling_results"
$csvFile = "$outputDir/telemetry_$timestamp.csv"
$reportFile = "$outputDir/report_$timestamp.txt"

# Create output directory
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

Write-Host "=== Hybrid Autoscaling Validation Test ===" -ForegroundColor Cyan
Write-Host "Parameters:"
Write-Host "  Users: $Users"
Write-Host "  Spawn Rate: $SpawnRate/s"
Write-Host "  Duration: $Duration seconds"
Write-Host "  Sample Interval: $SampleInterval seconds"
Write-Host "  Output: $csvFile"
Write-Host ""

# ==================================================
# Step 1: Pre-flight checks
# ==================================================
Write-Host "[1/6] Pre-flight checks..." -ForegroundColor Yellow

# Check if inference deployment exists
Write-Host "  → Checking inference deployment..."
$inference = kubectl get deployment inference -o json 2>$null | ConvertFrom-Json
if (-not $inference) {
    Write-Host "  ✗ Inference deployment not found!" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Inference deployment found" -ForegroundColor Green

# Check KEDA ScaledObject
Write-Host "  → Checking KEDA ScaledObject..."
$scaledObject = kubectl get scaledobject inference-slo-scaler -o json 2>$null | ConvertFrom-Json
if (-not $scaledObject) {
    Write-Host "  ✗ KEDA ScaledObject 'inference-slo-scaler' not found!" -ForegroundColor Red
    Write-Host "    Run: kubectl get scaledobject -A" -ForegroundColor Yellow
    exit 1
}
Write-Host "  ✓ KEDA ScaledObject found" -ForegroundColor Green
Write-Host "    Min replicas: $($scaledObject.spec.minReplicaCount)" -ForegroundColor Gray
Write-Host "    Max replicas: $($scaledObject.spec.maxReplicaCount)" -ForegroundColor Gray

# Check HPA
Write-Host "  → Checking HPA..."
$hpa = kubectl get hpa keda-hpa-inference-slo-scaler -o json 2>$null | ConvertFrom-Json
if (-not $hpa) {
    Write-Host "  ⚠ HPA 'keda-hpa-inference-slo-scaler' not found (optional)" -ForegroundColor Yellow
} else {
    Write-Host "  ✓ HPA found (managed by KEDA)" -ForegroundColor Green
    Write-Host "    Min replicas: $($hpa.spec.minReplicas)" -ForegroundColor Gray
    Write-Host "    Max replicas: $($hpa.spec.maxReplicas)" -ForegroundColor Gray
}

# Check Prometheus
Write-Host "  → Checking Prometheus..."
$promPod = kubectl get pods -l app.kubernetes.io/name=prometheus -o jsonpath='{.items[0].metadata.name}' 2>$null
if (-not $promPod) {
    Write-Host "  ✗ Prometheus pod not found!" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Prometheus pod: $promPod" -ForegroundColor Green

# Test Prometheus query
Write-Host "  → Testing Prometheus query..."
$testQuery = kubectl exec -n default $promPod -c prometheus-server -- wget -qO- 'http://localhost:9090/api/v1/query?query=up' 2>$null
if (-not $testQuery) {
    Write-Host "  ✗ Prometheus query failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Prometheus responding" -ForegroundColor Green

# Check Locust
Write-Host "  → Checking Locust master..."
$locustPod = kubectl get pods -l app=locust-master -o jsonpath='{.items[0].metadata.name}' 2>$null
if (-not $locustPod) {
    Write-Host "  ✗ Locust master not found!" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Locust master: $locustPod" -ForegroundColor Green

Write-Host ""

# ==================================================
# Step 2: Record baseline metrics
# ==================================================
Write-Host "[2/6] Recording baseline metrics..." -ForegroundColor Yellow

$baselineReplicas = (kubectl get pods -l app=inference --field-selector=status.phase=Running -o json | ConvertFrom-Json).items.Count
Write-Host "  Current replicas: $baselineReplicas" -ForegroundColor Cyan

# Initialize CSV
"timestamp,elapsed_sec,replicas,cpu_percent,mem_percent,p95_latency_ms,queue_len,notes" | Out-File -FilePath $csvFile -Encoding UTF8

Write-Host ""

# ==================================================
# Step 3: Start background telemetry collection
# ==================================================
Write-Host "[3/6] Starting telemetry collection..." -ForegroundColor Yellow

$telemetryJob = Start-Job -ScriptBlock {
    param($csvPath, $interval, $duration, $promPod)
    
    $startTime = Get-Date
    $endTime = $startTime.AddSeconds($duration + 30) # Extra buffer
    
    while ((Get-Date) -lt $endTime) {
        $now = Get-Date
        $elapsed = [math]::Round(($now - $startTime).TotalSeconds, 1)
        
        # Get replica count
        $replicas = 0
        try {
            $pods = kubectl get pods -l app=inference --field-selector=status.phase=Running -o json 2>$null | ConvertFrom-Json
            $replicas = $pods.items.Count
        } catch {}
        
        # Get CPU and Memory from top pods
        $cpu = 0
        $mem = 0
        try {
            $topOutput = kubectl top pods -l app=inference --no-headers 2>$null
            if ($topOutput) {
                $lines = $topOutput -split "`n" | Where-Object { $_ -match '\S' }
                $cpuValues = @()
                $memValues = @()
                foreach ($line in $lines) {
                    if ($line -match '(\d+)m\s+(\d+)Mi') {
                        $cpuValues += [int]$matches[1]
                        $memValues += [int]$matches[2]
                    }
                }
                if ($cpuValues.Count -gt 0) {
                    $cpu = [math]::Round(($cpuValues | Measure-Object -Average).Average, 0)
                    $mem = [math]::Round(($memValues | Measure-Object -Average).Average, 0)
                }
            }
        } catch {}
        
        # Get p95 latency from Prometheus
        $p95Latency = 0
        try {
            $query = 'histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[30s])) by (le))'
            $encodedQuery = [System.Web.HttpUtility]::UrlEncode($query)
            $promResult = kubectl exec -n default $promPod -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$encodedQuery" 2>$null
            if ($promResult) {
                $json = $promResult | ConvertFrom-Json
                if ($json.data.result.Count -gt 0) {
                    $p95Latency = [math]::Round([double]$json.data.result[0].value[1] * 1000, 2)
                }
            }
        } catch {}
        
        # Get queue length from Prometheus
        $queueLen = 0
        try {
            $query = 'avg(inference_queue_len)'
            $encodedQuery = [System.Web.HttpUtility]::UrlEncode($query)
            $promResult = kubectl exec -n default $promPod -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$encodedQuery" 2>$null
            if ($promResult) {
                $json = $promResult | ConvertFrom-Json
                if ($json.data.result.Count -gt 0) {
                    $queueLen = [math]::Round([double]$json.data.result[0].value[1], 2)
                }
            }
        } catch {}
        
        # Record sample
        $sample = "$($now.ToString('yyyy-MM-dd HH:mm:ss')),$elapsed,$replicas,$cpu,$mem,$p95Latency,$queueLen,collecting"
        $sample | Out-File -FilePath $csvPath -Append -Encoding UTF8
        
        Start-Sleep -Seconds $interval
    }
} -ArgumentList $csvFile, $SampleInterval, $Duration, $promPod

Write-Host "  ✓ Telemetry job started (PID: $($telemetryJob.Id))" -ForegroundColor Green
Write-Host ""

# ==================================================
# Step 4: Launch load test
# ==================================================
Write-Host "[4/6] Launching load test..." -ForegroundColor Yellow
Write-Host "  Target: http://inference:8000/predict" -ForegroundColor Cyan
Write-Host "  Users: $Users" -ForegroundColor Cyan
Write-Host "  Duration: $Duration seconds" -ForegroundColor Cyan

$loadTestStart = Get-Date

# Run Locust in background
$locustJob = Start-Job -ScriptBlock {
    param($pod, $users, $rate, $dur)
    kubectl exec -it deployment/locust-master -- locust `
        --headless `
        --users $users `
        --spawn-rate $rate `
        --run-time "$($dur)s" `
        --host http://inference:8000 `
        2>&1
} -ArgumentList $locustPod, $Users, $SpawnRate, $Duration

Write-Host "  ✓ Load test started (Job ID: $($locustJob.Id))" -ForegroundColor Green
Write-Host ""

# ==================================================
# Step 5: Monitor progress with live updates
# ==================================================
Write-Host "[5/6] Monitoring (updates every $SampleInterval seconds)..." -ForegroundColor Yellow
Write-Host ""

$progressInterval = $SampleInterval
$progressCount = [math]::Ceiling($Duration / $progressInterval)

for ($i = 0; $i -lt $progressCount; $i++) {
    $elapsed = $i * $progressInterval
    $remaining = $Duration - $elapsed
    
    # Read latest telemetry
    $latestLine = Get-Content $csvFile -Tail 1
    if ($latestLine -and $latestLine -ne "timestamp,elapsed_sec,replicas,cpu_percent,mem_percent,p95_latency_ms,queue_len,notes") {
        $fields = $latestLine -split ','
        if ($fields.Count -ge 7) {
            $time = $fields[0]
            $elap = $fields[1]
            $reps = $fields[2]
            $cpu = $fields[3]
            $mem = $fields[4]
            $p95 = $fields[5]
            $queue = $fields[6]
            
            Write-Host "[$elap`s] " -NoNewline -ForegroundColor Gray
            Write-Host "Replicas: " -NoNewline
            Write-Host "$reps " -NoNewline -ForegroundColor Cyan
            Write-Host "| CPU: " -NoNewline
            Write-Host "${cpu}m " -NoNewline -ForegroundColor $(if ([int]$cpu -gt 850) { "Red" } elseif ([int]$cpu -gt 600) { "Yellow" } else { "Green" })
            Write-Host "| Mem: " -NoNewline
            Write-Host "${mem}Mi " -NoNewline -ForegroundColor $(if ([int]$mem -gt 1600) { "Red" } elseif ([int]$mem -gt 1200) { "Yellow" } else { "Green" })
            Write-Host "| p95: " -NoNewline
            Write-Host "${p95}ms " -NoNewline -ForegroundColor $(if ([double]$p95 -gt 2000) { "Red" } elseif ([double]$p95 -gt 1500) { "Yellow" } else { "Green" })
            Write-Host "| Queue: " -NoNewline
            Write-Host "$queue" -ForegroundColor $(if ([double]$queue -gt 5) { "Red" } elseif ([double]$queue -gt 2) { "Yellow" } else { "Green" })
        }
    }
    
    if ($remaining -gt 0) {
        Start-Sleep -Seconds $progressInterval
    }
}

Write-Host ""
Write-Host "  ⏳ Waiting for load test to complete..." -ForegroundColor Yellow

# Wait for load test with timeout
$locustJob | Wait-Job -Timeout ($Duration + 30) | Out-Null

if ($locustJob.State -eq "Running") {
    Write-Host "  ⚠ Load test timed out, stopping..." -ForegroundColor Yellow
    Stop-Job -Job $locustJob
}

$locustOutput = Receive-Job -Job $locustJob 2>&1
Remove-Job -Job $locustJob -Force

Write-Host ""

# ==================================================
# Step 6: Stop telemetry and analyze results
# ==================================================
Write-Host "[6/6] Analyzing results..." -ForegroundColor Yellow

# Give telemetry a moment to catch up
Start-Sleep -Seconds 10

# Stop telemetry
Stop-Job -Job $telemetryJob
$telemetryOutput = Receive-Job -Job $telemetryJob
Remove-Job -Job $telemetryJob -Force

Write-Host "  ✓ Telemetry collection stopped" -ForegroundColor Green
Write-Host ""

# Parse CSV data
$samples = Import-Csv -Path $csvFile | Where-Object { $_.replicas -ne "" }

if ($samples.Count -eq 0) {
    Write-Host "  ✗ No telemetry data collected!" -ForegroundColor Red
    exit 1
}

Write-Host "  Samples collected: $($samples.Count)" -ForegroundColor Cyan
Write-Host ""

# Calculate statistics
$replicaCounts = $samples.replicas | ForEach-Object { [int]$_ }
$cpuValues = $samples.cpu_percent | Where-Object { $_ -ne "0" } | ForEach-Object { [int]$_ }
$memValues = $samples.mem_percent | Where-Object { $_ -ne "0" } | ForEach-Object { [int]$_ }
$latencyValues = $samples.p95_latency_ms | Where-Object { $_ -ne "0" } | ForEach-Object { [double]$_ }
$queueValues = $samples.queue_len | Where-Object { $_ -ne "0" } | ForEach-Object { [double]$_ }

# Replica statistics
$minReplicas = ($replicaCounts | Measure-Object -Minimum).Minimum
$maxReplicas = ($replicaCounts | Measure-Object -Maximum).Maximum
$avgReplicas = [math]::Round(($replicaCounts | Measure-Object -Average).Average, 1)

# CPU statistics
$avgCpu = if ($cpuValues.Count -gt 0) { [math]::Round(($cpuValues | Measure-Object -Average).Average, 0) } else { 0 }
$maxCpu = if ($cpuValues.Count -gt 0) { ($cpuValues | Measure-Object -Maximum).Maximum } else { 0 }

# Memory statistics
$avgMem = if ($memValues.Count -gt 0) { [math]::Round(($memValues | Measure-Object -Average).Average, 0) } else { 0 }
$maxMem = if ($memValues.Count -gt 0) { ($memValues | Measure-Object -Maximum).Maximum } else { 0 }

# Latency statistics
$avgLatency = if ($latencyValues.Count -gt 0) { [math]::Round(($latencyValues | Measure-Object -Average).Average, 2) } else { 0 }
$maxLatency = if ($latencyValues.Count -gt 0) { [math]::Round(($latencyValues | Measure-Object -Maximum).Maximum, 2) } else { 0 }

# Queue statistics
$avgQueue = if ($queueValues.Count -gt 0) { [math]::Round(($queueValues | Measure-Object -Average).Average, 2) } else { 0 }
$maxQueue = if ($queueValues.Count -gt 0) { [math]::Round(($queueValues | Measure-Object -Maximum).Maximum, 2) } else { 0 }

# Detect scaling events
$scalingEvents = @()
$prevReplicas = $baselineReplicas
foreach ($sample in $samples) {
    $currentReplicas = [int]$sample.replicas
    if ($currentReplicas -ne $prevReplicas) {
        $direction = if ($currentReplicas -gt $prevReplicas) { "SCALE UP" } else { "SCALE DOWN" }
        $scalingEvents += [PSCustomObject]@{
            Time = $sample.timestamp
            ElapsedSec = $sample.elapsed_sec
            Direction = $direction
            FromReplicas = $prevReplicas
            ToReplicas = $currentReplicas
            CPU = $sample.cpu_percent
            Latency = $sample.p95_latency_ms
            Queue = $sample.queue_len
        }
        $prevReplicas = $currentReplicas
    }
}

# Parse Locust results
$locustSummary = $locustOutput | Select-String -Pattern "Aggregated\s+(\d+)\s+(\d+)\(" | Select-Object -Last 1
$totalRequests = 0
$failedRequests = 0
$failureRate = 0

if ($locustSummary) {
    if ($locustSummary.Line -match 'Aggregated\s+(\d+)\s+(\d+)\(([0-9.]+)%\)') {
        $totalRequests = [int]$matches[1]
        $failedRequests = [int]$matches[2]
        $failureRate = [double]$matches[3]
    }
}

# ==================================================
# Generate Report
# ==================================================
$report = @"
=============================================================================
           HYBRID AUTOSCALING VALIDATION REPORT
=============================================================================
Test Configuration:
  Date/Time:         $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
  Users:             $Users
  Spawn Rate:        $SpawnRate/s
  Duration:          $Duration seconds
  Sample Interval:   $SampleInterval seconds
  Baseline Replicas: $baselineReplicas

=============================================================================
LOAD TEST RESULTS
=============================================================================
  Total Requests:    $totalRequests
  Failed Requests:   $failedRequests
  Failure Rate:      $failureRate%
  Success Rate:      $([math]::Round(100 - $failureRate, 2))%

=============================================================================
SCALING BEHAVIOR
=============================================================================
Replica Scaling:
  Min Replicas:      $minReplicas
  Max Replicas:      $maxReplicas
  Avg Replicas:      $avgReplicas
  Scaling Events:    $($scalingEvents.Count)

Resource Utilization:
  Avg CPU:           ${avgCpu}m
  Peak CPU:          ${maxCpu}m
  Avg Memory:        ${avgMem}Mi
  Peak Memory:       ${maxMem}Mi

Performance Metrics:
  Avg p95 Latency:   ${avgLatency}ms
  Peak p95 Latency:  ${maxLatency}ms
  Avg Queue Length:  $avgQueue
  Peak Queue Length: $maxQueue

=============================================================================
SCALING EVENTS TIMELINE
=============================================================================
"@

if ($scalingEvents.Count -gt 0) {
    $report += "`n"
    $report += $scalingEvents | Format-Table -AutoSize | Out-String
} else {
    $report += "`n  No scaling events detected during test period.`n"
}

$report += @"

=============================================================================
AUTOSCALER VALIDATION
=============================================================================
"@

# Check if KEDA triggered
$kedaTriggered = $false
if ($maxLatency -gt 1500 -or $maxQueue -gt 3) {
    $kedaTriggered = $true
    $report += "[OK] KEDA Triggers:`n"
    if ($maxLatency -gt 1500) {
        $report += "  - p95 Latency exceeded threshold (${maxLatency}ms > 1500ms)`n"
    }
    if ($maxQueue -gt 3) {
        $report += "  - Queue length exceeded threshold ($maxQueue > 3)`n"
    }
} else {
    $report += "[WARN] KEDA Triggers:`n"
    $report += "  - Latency/Queue did not exceed thresholds during test`n"
    $report += "    (Max latency: ${maxLatency}ms, Max queue: $maxQueue)`n"
}

# Check if HPA triggered
$hpaTriggered = $false
if ($maxCpu -gt 850 -or $maxMem -gt 1600) {
    $hpaTriggered = $true
    $report += "`n[OK] HPA Guardrails:`n"
    if ($maxCpu -gt 850) {
        $report += "  - CPU exceeded threshold (${maxCpu}m > 850m)`n"
    }
    if ($maxMem -gt 1600) {
        $report += "  - Memory exceeded threshold (${maxMem}Mi > 1600Mi)`n"
    }
} else {
    $report += "`n[WARN] HPA Guardrails:`n"
    $report += "  - CPU/Memory did not exceed thresholds during test`n"
    $report += "    (Max CPU: ${maxCpu}m, Max memory: ${maxMem}Mi)`n"
}

# Scaling responsiveness
if ($scalingEvents.Count -gt 0) {
    $firstScaleUp = $scalingEvents | Where-Object { $_.Direction -eq "SCALE UP" } | Select-Object -First 1
    if ($firstScaleUp) {
        $report += "`n[OK] Scaling Responsiveness:`n"
        $report += "  - First scale-up at: $($firstScaleUp.ElapsedSec)s`n"
        $report += "  - Triggered by: Latency ${firstScaleUp.Latency}ms, Queue $($firstScaleUp.Queue)`n"
    }
}

$report += @"

=============================================================================
RECOMMENDATIONS
=============================================================================
"@

$recommendations = @()

if ($maxReplicas -eq $minReplicas) {
    $recommendations += "[WARN] No scaling occurred - consider:"
    $recommendations += "  - Increasing load (more users or longer duration)"
    $recommendations += "  - Lowering KEDA/HPA thresholds"
    $recommendations += "  - Checking autoscaler configuration"
}

if ($failureRate -gt 1) {
    $recommendations += "[WARN] High failure rate detected ($failureRate%)"
    $recommendations += "  - Increase maxReplicas to handle peak load"
    $recommendations += "  - Reduce latency thresholds for faster scaling"
}

if ($maxLatency -gt 3000) {
    $recommendations += "[WARN] High latency detected (${maxLatency}ms)"
    $recommendations += "  - Consider aggressive scaling triggers"
    $recommendations += "  - Review inference service performance"
}

if ($scalingEvents.Count -eq 0) {
    $recommendations += "[INFO] Stable replica count throughout test"
    $recommendations += "  - Workload was within capacity of initial replicas"
    $recommendations += "  - For stress testing, increase load parameters"
}

if ($recommendations.Count -eq 0) {
    $recommendations += "[OK] Autoscaling performed as expected"
    $recommendations += "  - KEDA and HPA working correctly"
    $recommendations += "  - No tuning required at this time"
}

$report += ($recommendations -join "`n") + "`n"

$report += @"

=============================================================================
DATA FILES
=============================================================================
  Telemetry CSV:  $csvFile
  Full Report:    $reportFile

=============================================================================
"@

# Save report
$report | Out-File -FilePath $reportFile -Encoding UTF8

# Display report
Write-Host $report

# Display key telemetry samples
Write-Host ""
Write-Host "=== KEY TELEMETRY SAMPLES (First 10 & Last 10) ===" -ForegroundColor Cyan
Write-Host ""
$samples | Select-Object -First 10 | Format-Table -AutoSize
Write-Host "..." -ForegroundColor Gray
$samples | Select-Object -Last 10 | Format-Table -AutoSize

Write-Host ""
Write-Host "=== TEST COMPLETE ===" -ForegroundColor Green
Write-Host "Full results saved to: $outputDir" -ForegroundColor Cyan
Write-Host "  - Telemetry: $csvFile" -ForegroundColor Gray
Write-Host "  - Report: $reportFile" -ForegroundColor Gray
Write-Host ""

# Return exit code based on success criteria
if ($failureRate -gt 5) {
    Write-Host "[WARN] Warning: High failure rate detected!" -ForegroundColor Yellow
    exit 1
}

exit 0
