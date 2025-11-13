#!/usr/bin/env pwsh
# Real-time Hybrid Autoscaling Validation Test
# Collects telemetry every 5 seconds while running load test

param(
    [int]$Users = 150,
    [int]$SpawnRate = 10,
    [int]$Duration = 180,
    [int]$SampleInterval = 5
)

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputDir = Join-Path $scriptDir "autoscaling_results"
$csvFile = Join-Path $outputDir "telemetry_$timestamp.csv"
$reportFile = Join-Path $outputDir "report_$timestamp.txt"

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
Write-Host ""

# Pre-flight checks
Write-Host "[1/5] Pre-flight checks..." -ForegroundColor Yellow

# Check inference deployment
Write-Host "  -> Checking inference deployment..."
$inference = kubectl get deployment inference -o json 2>$null | ConvertFrom-Json
if (-not $inference) {
    Write-Host "  ERROR: Inference deployment not found!" -ForegroundColor Red
    exit 1
}
Write-Host "  OK: Inference deployment found" -ForegroundColor Green

# Check KEDA ScaledObject
Write-Host "  -> Checking KEDA ScaledObject..."
$scaledObject = kubectl get scaledobject inference-slo-scaler -o json 2>$null | ConvertFrom-Json
if (-not $scaledObject) {
    Write-Host "  ERROR: KEDA ScaledObject not found!" -ForegroundColor Red
    exit 1
}
$minRep = $scaledObject.spec.minReplicaCount
$maxRep = $scaledObject.spec.maxReplicaCount
Write-Host "  OK: KEDA ScaledObject found (min: $minRep, max: $maxRep)" -ForegroundColor Green

# Check Prometheus
Write-Host "  -> Checking Prometheus..."
$promPod = kubectl get pods -l app.kubernetes.io/name=prometheus -o jsonpath='{.items[0].metadata.name}' 2>$null
if (-not $promPod) {
    Write-Host "  ERROR: Prometheus pod not found!" -ForegroundColor Red
    exit 1
}
Write-Host "  OK: Prometheus pod: $promPod" -ForegroundColor Green

# Check Locust
Write-Host "  -> Checking Locust master..."
$locustPod = kubectl get pods -l role=master -o jsonpath='{.items[0].metadata.name}' 2>$null
if (-not $locustPod) {
    Write-Host "  ERROR: Locust master not found!" -ForegroundColor Red
    exit 1
}
Write-Host "  OK: Locust master: $locustPod" -ForegroundColor Green
Write-Host ""

# Record baseline
Write-Host "[2/5] Recording baseline..." -ForegroundColor Yellow
$baselineReplicas = (kubectl get pods -l app=inference --field-selector=status.phase=Running -o json | ConvertFrom-Json).items.Count
Write-Host "  Current replicas: $baselineReplicas" -ForegroundColor Cyan
Write-Host ""

# Initialize CSV
"timestamp,elapsed_sec,replicas,cpu_m,mem_mi,p95_latency_ms,queue_len" | Out-File -FilePath $csvFile -Encoding UTF8

# Start telemetry collection job
Write-Host "[3/5] Starting telemetry collection..." -ForegroundColor Yellow

$telemetryScript = {
    param($csvPath, $interval, $duration, $promPod)
    
    $startTime = Get-Date
    $endTime = $startTime.AddSeconds($duration + 30)
    
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
            $query = 'histogram_quantile(0.95,sum(rate(inference_latency_seconds_bucket[30s]))by(le))'
            $promResult = kubectl exec -n default $promPod -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$query" 2>$null
            if ($promResult) {
                $json = $promResult | ConvertFrom-Json
                if ($json.data.result.Count -gt 0) {
                    $p95Latency = [math]::Round([double]$json.data.result[0].value[1] * 1000, 2)
                }
            }
        } catch {}
        
        # Get queue length
        $queueLen = 0
        try {
            $query = 'avg(inference_queue_len)'
            $promResult = kubectl exec -n default $promPod -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$query" 2>$null
            if ($promResult) {
                $json = $promResult | ConvertFrom-Json
                if ($json.data.result.Count -gt 0) {
                    $queueLen = [math]::Round([double]$json.data.result[0].value[1], 2)
                }
            }
        } catch {}
        
        # Record sample
        $sample = "$($now.ToString('yyyy-MM-dd HH:mm:ss')),$elapsed,$replicas,$cpu,$mem,$p95Latency,$queueLen"
        $sample | Out-File -FilePath $csvPath -Append -Encoding UTF8
        
        Start-Sleep -Seconds $interval
    }
}

$telemetryJob = Start-Job -ScriptBlock $telemetryScript -ArgumentList $csvFile, $SampleInterval, $Duration, $promPod
Write-Host "  OK: Telemetry job started (Job ID: $($telemetryJob.Id))" -ForegroundColor Green
Write-Host ""

# Launch load test
Write-Host "[4/5] Launching load test..." -ForegroundColor Yellow
Write-Host "  Target: http://inference:8000/predict" -ForegroundColor Cyan
Write-Host "  Users: $Users | Duration: ${Duration}s" -ForegroundColor Cyan

$loadTestStart = Get-Date

$locustScript = {
    param($pod, $users, $rate, $dur)
    kubectl exec -it deployment/locust-master -- locust --headless --users $users --spawn-rate $rate --run-time "$($dur)s" --host http://inference:8000 2>&1
}

$locustJob = Start-Job -ScriptBlock $locustScript -ArgumentList $locustPod, $Users, $SpawnRate, $Duration
Write-Host "  OK: Load test started (Job ID: $($locustJob.Id))" -ForegroundColor Green
Write-Host ""

# Monitor progress with live updates
Write-Host "[5/5] Monitoring (updates every $SampleInterval seconds)..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Time | Replicas | CPU(m) | Mem(Mi) | p95(ms) | Queue" -ForegroundColor Cyan
Write-Host "-----+----------+--------+---------+---------+-------" -ForegroundColor Cyan

$progressInterval = $SampleInterval
$progressCount = [math]::Ceiling($Duration / $progressInterval)

for ($i = 0; $i -lt $progressCount; $i++) {
    # Read latest telemetry
    Start-Sleep -Seconds $progressInterval
    
    $latestLine = Get-Content $csvFile -Tail 1 -ErrorAction SilentlyContinue
    if ($latestLine -and $latestLine -notmatch '^timestamp') {
        $fields = $latestLine -split ','
        if ($fields.Count -ge 7) {
            $time = ([datetime]$fields[0]).ToString('HH:mm:ss')
            $elap = $fields[1]
            $reps = $fields[2]
            $cpu = $fields[3]
            $mem = $fields[4]
            $p95 = $fields[5]
            $queue = $fields[6]
            
            Write-Host "$time | " -NoNewline -ForegroundColor Gray
            Write-Host "$reps".PadRight(8) -NoNewline -ForegroundColor $(if ([int]$reps -gt 3) { "Yellow" } else { "Cyan" })
            Write-Host " | $cpu".PadRight(6) -NoNewline -ForegroundColor $(if ([int]$cpu -gt 850) { "Red" } elseif ([int]$cpu -gt 600) { "Yellow" } else { "Green" })
            Write-Host " | $mem".PadRight(7) -NoNewline -ForegroundColor $(if ([int]$mem -gt 1600) { "Red" } elseif ([int]$mem -gt 1200) { "Yellow" } else { "Green" })
            Write-Host " | $p95".PadRight(7) -NoNewline -ForegroundColor $(if ([double]$p95 -gt 2000) { "Red" } elseif ([double]$p95 -gt 1500) { "Yellow" } else { "Green" })
            Write-Host " | $queue" -ForegroundColor $(if ([double]$queue -gt 5) { "Red" } elseif ([double]$queue -gt 2) { "Yellow" } else { "Green" })
        }
    }
}

Write-Host ""
Write-Host "  Waiting for load test to complete..." -ForegroundColor Yellow

# Wait for load test
$locustJob | Wait-Job -Timeout ($Duration + 30) | Out-Null
if ($locustJob.State -eq "Running") {
    Stop-Job -Job $locustJob
}
$locustOutput = Receive-Job -Job $locustJob 2>&1
Remove-Job -Job $locustJob -Force

# Stop telemetry
Start-Sleep -Seconds 10
Stop-Job -Job $telemetryJob
Receive-Job -Job $telemetryJob | Out-Null
Remove-Job -Job $telemetryJob -Force

Write-Host "  OK: Collection stopped" -ForegroundColor Green
Write-Host ""

# Analyze results
Write-Host "=== Analyzing Results ===" -ForegroundColor Cyan
Write-Host ""

$samples = Import-Csv -Path $csvFile | Where-Object { $_.replicas -ne "" }

if ($samples.Count -eq 0) {
    Write-Host "ERROR: No telemetry data collected!" -ForegroundColor Red
    exit 1
}

Write-Host "Samples collected: $($samples.Count)" -ForegroundColor Cyan
Write-Host ""

# Calculate statistics
$replicaCounts = $samples.replicas | ForEach-Object { [int]$_ }
$cpuValues = $samples.cpu_m | Where-Object { $_ -ne "0" } | ForEach-Object { [int]$_ }
$memValues = $samples.mem_mi | Where-Object { $_ -ne "0" } | ForEach-Object { [int]$_ }
$latencyValues = $samples.p95_latency_ms | Where-Object { $_ -ne "0" } | ForEach-Object { [double]$_ }
$queueValues = $samples.queue_len | Where-Object { $_ -ne "0" } | ForEach-Object { [double]$_ }

$minReplicas = ($replicaCounts | Measure-Object -Minimum).Minimum
$maxReplicas = ($replicaCounts | Measure-Object -Maximum).Maximum
$avgReplicas = [math]::Round(($replicaCounts | Measure-Object -Average).Average, 1)

$avgCpu = if ($cpuValues.Count -gt 0) { [math]::Round(($cpuValues | Measure-Object -Average).Average, 0) } else { 0 }
$maxCpu = if ($cpuValues.Count -gt 0) { ($cpuValues | Measure-Object -Maximum).Maximum } else { 0 }

$avgMem = if ($memValues.Count -gt 0) { [math]::Round(($memValues | Measure-Object -Average).Average, 0) } else { 0 }
$maxMem = if ($memValues.Count -gt 0) { ($memValues | Measure-Object -Maximum).Maximum } else { 0 }

$avgLatency = if ($latencyValues.Count -gt 0) { [math]::Round(($latencyValues | Measure-Object -Average).Average, 2) } else { 0 }
$maxLatency = if ($latencyValues.Count -gt 0) { [math]::Round(($latencyValues | Measure-Object -Maximum).Maximum, 2) } else { 0 }

$avgQueue = if ($queueValues.Count -gt 0) { [math]::Round(($queueValues | Measure-Object -Average).Average, 2) } else { 0 }
$maxQueue = if ($queueValues.Count -gt 0) { [math]::Round(($queueValues | Measure-Object -Maximum).Maximum, 2) } else { 0 }

# Detect scaling events
$scalingEvents = @()
$prevReplicas = $baselineReplicas
foreach ($sample in $samples) {
    $currentReplicas = [int]$sample.replicas
    if ($currentReplicas -ne $prevReplicas) {
        $direction = if ($currentReplicas -gt $prevReplicas) { "SCALE_UP" } else { "SCALE_DOWN" }
        $scalingEvents += [PSCustomObject]@{
            Time = $sample.timestamp
            ElapsedSec = $sample.elapsed_sec
            Direction = $direction
            FromReplicas = $prevReplicas
            ToReplicas = $currentReplicas
            CPU_m = $sample.cpu_m
            Latency_ms = $sample.p95_latency_ms
            Queue = $sample.queue_len
        }
        $prevReplicas = $currentReplicas
    }
}

# Parse Locust results
$locustSummary = $locustOutput | Select-String -Pattern 'Aggregated\s+(\d+)\s+(\d+)\(' | Select-Object -Last 1
$totalRequests = 0
$failedRequests = 0
$failureRate = 0

if ($locustSummary) {
    if ($locustSummary.Line -match 'Aggregated\s+(\d+)\s+(\d+)\(([0-9.]+)') {
        $totalRequests = [int]$matches[1]
        $failedRequests = [int]$matches[2]
        $failureRate = [double]$matches[3]
    }
}

# Display summary
Write-Host "=== RESULTS SUMMARY ===" -ForegroundColor Yellow
Write-Host ""
Write-Host "Load Test:" -ForegroundColor Cyan
Write-Host "  Total Requests:    $totalRequests"
Write-Host "  Failed Requests:   $failedRequests"
Write-Host "  Failure Rate:      $failureRate%"
Write-Host ""

Write-Host "Replica Scaling:" -ForegroundColor Cyan
Write-Host "  Min Replicas:      $minReplicas"
Write-Host "  Max Replicas:      $maxReplicas"
Write-Host "  Avg Replicas:      $avgReplicas"
Write-Host "  Scaling Events:    $($scalingEvents.Count)"
Write-Host ""

Write-Host "Resource Utilization:" -ForegroundColor Cyan
Write-Host "  Avg CPU:           ${avgCpu}m"
Write-Host "  Peak CPU:          ${maxCpu}m"
Write-Host "  Avg Memory:        ${avgMem}Mi"
Write-Host "  Peak Memory:       ${maxMem}Mi"
Write-Host ""

Write-Host "Performance Metrics:" -ForegroundColor Cyan
Write-Host "  Avg p95 Latency:   ${avgLatency}ms"
Write-Host "  Peak p95 Latency:  ${maxLatency}ms"
Write-Host "  Avg Queue Length:  $avgQueue"
Write-Host "  Peak Queue Length: $maxQueue"
Write-Host ""

if ($scalingEvents.Count -gt 0) {
    Write-Host "=== SCALING EVENTS ===" -ForegroundColor Yellow
    Write-Host ""
    $scalingEvents | Format-Table -AutoSize
    Write-Host ""
}

# Validation checks
Write-Host "=== AUTOSCALER VALIDATION ===" -ForegroundColor Yellow
Write-Host ""

# KEDA check
if ($maxLatency -gt 500 -or $maxQueue -gt 20) {
    Write-Host "[OK] KEDA Prometheus triggers activated:" -ForegroundColor Green
    if ($maxLatency -gt 500) {
        Write-Host "  - p95 Latency: ${maxLatency}ms (threshold: 500ms)"
    }
    if ($maxQueue -gt 20) {
        Write-Host "  - Queue length: $maxQueue (threshold: 20)"
    }
} else {
    Write-Host "[INFO] KEDA Prometheus triggers below thresholds:" -ForegroundColor Cyan
    Write-Host "  - Max latency: ${maxLatency}ms (threshold: 500ms)"
    Write-Host "  - Max queue: $maxQueue (threshold: 20)"
}
Write-Host ""

# HPA check
if ($maxCpu -gt 850 -or $maxMem -gt 1600) {
    Write-Host "[OK] HPA resource guardrails activated:" -ForegroundColor Green
    if ($maxCpu -gt 850) {
        Write-Host "  - CPU: ${maxCpu}m (threshold: 850m / 85%)"
    }
    if ($maxMem -gt 1600) {
        Write-Host "  - Memory: ${maxMem}Mi (threshold: 80%)"
    }
} else {
    Write-Host "[INFO] HPA resource guardrails below thresholds:" -ForegroundColor Cyan
    Write-Host "  - Max CPU: ${maxCpu}m (threshold: 850m / 85%)"
    Write-Host "  - Max memory: ${maxMem}Mi"
}
Write-Host ""

# Scaling behavior
if ($scalingEvents.Count -gt 0) {
    Write-Host "[OK] Scaling behavior observed:" -ForegroundColor Green
    $firstScaleUp = $scalingEvents | Where-Object { $_.Direction -eq "SCALE_UP" } | Select-Object -First 1
    if ($firstScaleUp) {
        Write-Host "  - First scale-up at: $($firstScaleUp.ElapsedSec)s"
        Write-Host "  - Scaled from $($firstScaleUp.FromReplicas) to $($firstScaleUp.ToReplicas) replicas"
    }
} else {
    Write-Host "[INFO] No scaling events detected" -ForegroundColor Cyan
    Write-Host "  - Workload within capacity of initial $baselineReplicas replicas"
}
Write-Host ""

# Recommendations
Write-Host "=== RECOMMENDATIONS ===" -ForegroundColor Yellow
Write-Host ""

if ($maxReplicas -eq $minReplicas -and $maxLatency -lt 500) {
    Write-Host "[INFO] System handled load without scaling"
    Write-Host "  -> Consider higher load to test autoscaling behavior"
} elseif ($failureRate -gt 1) {
    Write-Host "[WARN] High failure rate detected"
    Write-Host "  -> Increase maxReplicas or lower latency thresholds"
} elseif ($maxLatency -gt 3000) {
    Write-Host "[WARN] High latency detected"
    Write-Host "  -> Review inference service performance"
    Write-Host "  -> Consider more aggressive scaling triggers"
} else {
    Write-Host "[OK] Autoscaling performed as expected"
    Write-Host "  -> KEDA and HPA working correctly"
}
Write-Host ""

# Save results
$report = @"
HYBRID AUTOSCALING VALIDATION REPORT
Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

Test Configuration:
  Users: $Users
  Spawn Rate: $SpawnRate/s
  Duration: $Duration seconds
  Baseline Replicas: $baselineReplicas

Load Test Results:
  Total Requests: $totalRequests
  Failed Requests: $failedRequests
  Failure Rate: $failureRate%

Scaling Behavior:
  Min Replicas: $minReplicas
  Max Replicas: $maxReplicas
  Avg Replicas: $avgReplicas
  Scaling Events: $($scalingEvents.Count)

Resource Utilization:
  Avg CPU: ${avgCpu}m
  Peak CPU: ${maxCpu}m
  Avg Memory: ${avgMem}Mi
  Peak Memory: ${maxMem}Mi

Performance Metrics:
  Avg p95 Latency: ${avgLatency}ms
  Peak p95 Latency: ${maxLatency}ms
  Avg Queue Length: $avgQueue
  Peak Queue Length: $maxQueue

Data Files:
  Telemetry CSV: $csvFile
  Full Report: $reportFile
"@

$report | Out-File -FilePath $reportFile -Encoding UTF8

Write-Host "=== TEST COMPLETE ===" -ForegroundColor Green
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Cyan
Write-Host "  - Telemetry CSV: $csvFile"
Write-Host "  - Report: $reportFile"
Write-Host ""

# Display first and last samples
Write-Host "Sample Telemetry (First 5 rows):" -ForegroundColor Cyan
$samples | Select-Object -First 5 | Format-Table -AutoSize
Write-Host "Sample Telemetry (Last 5 rows):" -ForegroundColor Cyan
$samples | Select-Object -Last 5 | Format-Table -AutoSize

exit 0
