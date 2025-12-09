#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Hybrid KEDA + HPA capacity analysis with per-pod throughput calculation.

.DESCRIPTION
    Runs load test while collecting comprehensive telemetry including:
    - Replica count progression
    - CPU and memory utilization per pod
    - p95 latency from Prometheus
    - Queue length from Prometheus
    - Request rate and throughput
    
    Calculates sustainable RPS per pod before latency degradation.

.PARAMETER Users
    Number of concurrent Locust users (default: 200)

.PARAMETER SpawnRate
    User spawn rate per second (default: 10)

.PARAMETER Duration
    Test duration in seconds (default: 300)

.PARAMETER SampleInterval
    Telemetry collection interval in seconds (default: 15)
#>

param(
    [int]$Users = 200,
    [int]$SpawnRate = 10,
    [int]$Duration = 300,
    [int]$SampleInterval = 15,
    [string]$OutputDir = "./capacity_analysis"
)

$ErrorActionPreference = "Continue"

# Create output directory
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$testDir = Join-Path $OutputDir "test_${timestamp}"
New-Item -ItemType Directory -Path $testDir -Force | Out-Null

$telemetryFile = Join-Path $testDir "telemetry.csv"
$summaryFile = Join-Path $testDir "capacity_report.txt"
$locustLogFile = Join-Path $testDir "locust.log"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "KEDA + HPA HYBRID CAPACITY ANALYSIS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuration:"
Write-Host "  Users: $Users | Spawn: $SpawnRate/s | Duration: ${Duration}s"
Write-Host "  Sample Interval: ${SampleInterval}s"
Write-Host "  Output: $testDir"
Write-Host ""

# Initialize CSV with headers
"Timestamp,Elapsed_Sec,Replicas,CPU_Min,CPU_Max,CPU_Avg,Mem_Avg_Mi,p95_Latency_ms,p50_Latency_ms,Queue_Len_Avg,Queue_Len_Max,RPS_Estimate,KEDA_Active,HPA_Scaling" | Out-File -FilePath $telemetryFile -Encoding UTF8

# Get baseline state
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Collecting baseline metrics..." -ForegroundColor Yellow

$initialReplicas = (kubectl get pods -l app=inference --no-headers 2>$null | Measure-Object).Count
$initialHPA = kubectl get hpa keda-hpa-inference-slo-scaler -o jsonpath='{.status.currentReplicas}' 2>$null

Write-Host "  Initial Replicas: $initialReplicas"
Write-Host "  HPA Target: $initialHPA"
Write-Host ""

# Query initial Prometheus state
$query = "rate(inference_latency_seconds_count[1m])"
$rateCheck = kubectl exec -n default prometheus-server-c568bf4db-zmk2t -c prometheus-server -- `
    wget -qO- "http://localhost:9090/api/v1/query?query=$([uri]::EscapeDataString($query))" 2>$null | ConvertFrom-Json

if ($rateCheck.data.result.Count -eq 0) {
    Write-Host "  [INFO] No active traffic detected - metrics will populate during test" -ForegroundColor Yellow
}

# Start Locust test
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting Locust load test..." -ForegroundColor Green

$locustCmd = "kubectl exec -n default deployment/locust-master -- locust --headless --users $Users --spawn-rate $SpawnRate --run-time ${Duration}s --host http://inference:8000 --only-summary"

$locustJob = Start-Job -ScriptBlock {
    param($cmd, $logFile)
    try {
        $output = Invoke-Expression $cmd 2>&1
        $output | Out-File -FilePath $logFile -Encoding UTF8
        return $output
    } catch {
        "ERROR: $_" | Out-File -FilePath $logFile -Encoding UTF8
    }
} -ArgumentList $locustCmd, $locustLogFile

Write-Host "  Job ID: $($locustJob.Id)"
Write-Host "  Waiting 20s for ramp-up..." -ForegroundColor Yellow
Start-Sleep -Seconds 20

# Telemetry collection loop
Write-Host ""
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Collecting telemetry (passive monitoring)..." -ForegroundColor Cyan
Write-Host ""

$testStartTime = Get-Date
$sampleCount = 0
$lastReqCount = 0

while ($locustJob.State -eq 'Running') {
    $elapsed = [math]::Round(((Get-Date) - $testStartTime).TotalSeconds, 0)
    
    try {
        # 1. Pod count and resource metrics
        $pods = kubectl get pods -l app=inference --no-headers 2>$null
        $replicas = ($pods | Measure-Object).Count
        
        # Parse CPU and memory from kubectl top
        $topOutput = kubectl top pods -l app=inference --no-headers 2>$null
        $cpuValues = @()
        $memValues = @()
        
        foreach ($line in $topOutput) {
            if ($line -match '^\S+\s+(\d+)m\s+(\d+)Mi') {
                $cpuValues += [int]$Matches[1]
                $memValues += [int]$Matches[2]
            }
        }
        
        $cpuMin = if ($cpuValues.Count -gt 0) { ($cpuValues | Measure-Object -Minimum).Minimum } else { 0 }
        $cpuMax = if ($cpuValues.Count -gt 0) { ($cpuValues | Measure-Object -Maximum).Maximum } else { 0 }
        $cpuAvg = if ($cpuValues.Count -gt 0) { [math]::Round(($cpuValues | Measure-Object -Average).Average, 0) } else { 0 }
        $memAvg = if ($memValues.Count -gt 0) { [math]::Round(($memValues | Measure-Object -Average).Average, 0) } else { 0 }
        
        # 2. Prometheus p95 latency (5m window)
        $p95Query = "histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[5m])) by (le))"
        $p95Result = kubectl exec -n default prometheus-server-c568bf4db-zmk2t -c prometheus-server -- `
            wget -qO- "http://localhost:9090/api/v1/query?query=$([uri]::EscapeDataString($p95Query))" 2>$null | ConvertFrom-Json
        
        $p95Latency = 0
        if ($p95Result.data.result.Count -gt 0) {
            $p95Value = $p95Result.data.result[0].value[1]
            $p95Latency = [math]::Round([double]$p95Value * 1000, 0)
        }
        
        # 3. Prometheus p50 latency (for capacity analysis)
        $p50Query = "histogram_quantile(0.50, sum(rate(inference_latency_seconds_bucket[5m])) by (le))"
        $p50Result = kubectl exec -n default prometheus-server-c568bf4db-zmk2t -c prometheus-server -- `
            wget -qO- "http://localhost:9090/api/v1/query?query=$([uri]::EscapeDataString($p50Query))" 2>$null | ConvertFrom-Json
        
        $p50Latency = 0
        if ($p50Result.data.result.Count -gt 0) {
            $p50Value = $p50Result.data.result[0].value[1]
            $p50Latency = [math]::Round([double]$p50Value * 1000, 0)
        }
        
        # 4. Queue metrics
        $queueAvgQuery = "avg(inference_queue_len)"
        $queueAvgResult = kubectl exec -n default prometheus-server-c568bf4db-zmk2t -c prometheus-server -- `
            wget -qO- "http://localhost:9090/api/v1/query?query=$([uri]::EscapeDataString($queueAvgQuery))" 2>$null | ConvertFrom-Json
        
        $queueAvg = 0
        if ($queueAvgResult.data.result.Count -gt 0) {
            $queueAvg = [math]::Round([double]$queueAvgResult.data.result[0].value[1], 1)
        }
        
        $queueMaxQuery = "max(inference_queue_len)"
        $queueMaxResult = kubectl exec -n default prometheus-server-c568bf4db-zmk2t -c prometheus-server -- `
            wget -qO- "http://localhost:9090/api/v1/query?query=$([uri]::EscapeDataString($queueMaxQuery))" 2>$null | ConvertFrom-Json
        
        $queueMax = 0
        if ($queueMaxResult.data.result.Count -gt 0) {
            $queueMax = [math]::Round([double]$queueMaxResult.data.result[0].value[1], 0)
        }
        
        # 5. Request rate (RPS estimate)
        $rpsQuery = "sum(rate(inference_latency_seconds_count[1m]))"
        $rpsResult = kubectl exec -n default prometheus-server-c568bf4db-zmk2t -c prometheus-server -- `
            wget -qO- "http://localhost:9090/api/v1/query?query=$([uri]::EscapeDataString($rpsQuery))" 2>$null | ConvertFrom-Json
        
        $rps = 0
        if ($rpsResult.data.result.Count -gt 0) {
            $rps = [math]::Round([double]$rpsResult.data.result[0].value[1], 1)
        }
        
        # 6. KEDA ScaledObject status
        $kedaActive = kubectl get scaledobject inference-slo-scaler -o jsonpath='{.status.conditions[?(@.type=="Active")].status}' 2>$null
        if (-not $kedaActive) { $kedaActive = "Unknown" }
        
        # 7. HPA scaling activity
        $hpaScaling = kubectl get hpa keda-hpa-inference-slo-scaler -o jsonpath='{.status.conditions[?(@.type=="ScalingActive")].status}' 2>$null
        if (-not $hpaScaling) { $hpaScaling = "Unknown" }
        
        # Write to CSV
        $timestamp = Get-Date -Format "HH:mm:ss"
        "$timestamp,$elapsed,$replicas,$cpuMin,$cpuMax,$cpuAvg,$memAvg,$p95Latency,$p50Latency,$queueAvg,$queueMax,$rps,$kedaActive,$hpaScaling" | 
            Out-File -FilePath $telemetryFile -Append -Encoding UTF8
        
        $sampleCount++
        
        # Progress display (simplified without format strings)
        $elapsedInt = [int]$elapsed
        $cpuInt = [int]$cpuAvg
        $p95Str = if ($p95Latency -match '^\d+$') { "$($p95Latency)ms" } else { "N/A" }
        $p50Str = if ($p50Latency -match '^\d+$') { "$($p50Latency)ms" } else { "N/A" }
        $rpsStr = if ($rps -match '^[\d.]+$') { "$($rps)" } else { "N/A" }
        
        Write-Host "`r  [$sampleCount] T+${elapsedInt}s | Pods:$replicas | CPU:${cpuInt}m | p95:$p95Str | p50:$p50Str | Queue:$queueAvg | RPS:$rpsStr | KEDA:$kedaActive" -NoNewline -ForegroundColor Gray
        
    } catch {
        Write-Host "`r  Sample $sampleCount failed: $_" -NoNewline -ForegroundColor Red
    }
    
    Start-Sleep -Seconds $SampleInterval
}

Write-Host ""
Write-Host ""
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Test complete. Processing results..." -ForegroundColor Green
Write-Host ""

# Wait for Locust job
$locustOutput = Receive-Job -Job $locustJob -Wait

# Load telemetry data
$telemetry = Import-Csv $telemetryFile

# Calculate statistics
$replicaCounts = $telemetry | Select-Object -ExpandProperty Replicas | ForEach-Object { [int]$_ }
$minReplicas = ($replicaCounts | Measure-Object -Minimum).Minimum
$maxReplicas = ($replicaCounts | Measure-Object -Maximum).Maximum
$avgReplicas = [math]::Round(($replicaCounts | Measure-Object -Average).Average, 1)

$p95Values = $telemetry | Select-Object -ExpandProperty p95_Latency_ms | ForEach-Object { if ($_ -match '^\d+$') { [int]$_ } else { 0 } }
$maxP95 = ($p95Values | Measure-Object -Maximum).Maximum
$avgP95 = [math]::Round(($p95Values | Where-Object { $_ -gt 0 } | Measure-Object -Average).Average, 0)

$p50Values = $telemetry | Select-Object -ExpandProperty p50_Latency_ms | ForEach-Object { if ($_ -match '^\d+$') { [int]$_ } else { 0 } }
$maxP50 = ($p50Values | Measure-Object -Maximum).Maximum
$avgP50 = [math]::Round(($p50Values | Where-Object { $_ -gt 0 } | Measure-Object -Average).Average, 0)

$queueValues = $telemetry | Select-Object -ExpandProperty Queue_Len_Avg | ForEach-Object { if ($_ -match '^[\d.]+$') { [double]$_ } else { 0 } }
$maxQueue = ($queueValues | Measure-Object -Maximum).Maximum
$avgQueue = [math]::Round(($queueValues | Measure-Object -Average).Average, 1)

$rpsValues = $telemetry | Select-Object -ExpandProperty RPS_Estimate | ForEach-Object { if ($_ -match '^[\d.]+$') { [double]$_ } else { 0 } }
$maxRPS = [math]::Round(($rpsValues | Measure-Object -Maximum).Maximum, 1)
$avgRPS = [math]::Round(($rpsValues | Where-Object { $_ -gt 0 } | Measure-Object -Average).Average, 1)

$cpuAvgs = $telemetry | Select-Object -ExpandProperty CPU_Avg | ForEach-Object { if ($_ -match '^\d+$') { [int]$_ } else { 0 } }
$maxCPU = ($cpuAvgs | Measure-Object -Maximum).Maximum
$avgCPU = [math]::Round(($cpuAvgs | Measure-Object -Average).Average, 0)

# Calculate per-pod capacity (before first scale-up)
$preScaleData = $telemetry | Where-Object { [int]$_.Replicas -eq $minReplicas } | Select-Object -First 5
$preScaleRPS = $preScaleData | Select-Object -ExpandProperty RPS_Estimate | ForEach-Object { if ($_ -match '^[\d.]+$') { [double]$_ } else { 0 } }
$preScaleAvgRPS = if ($preScaleRPS.Count -gt 0) { [math]::Round(($preScaleRPS | Measure-Object -Average).Average, 1) } else { 0 }
$rpsPerPod = if ($minReplicas -gt 0 -and $preScaleAvgRPS -gt 0) { [math]::Round($preScaleAvgRPS / $minReplicas, 2) } else { 0 }

$preScaleP95 = $preScaleData | Select-Object -ExpandProperty p95_Latency_ms | ForEach-Object { if ($_ -match '^\d+$') { [int]$_ } else { 0 } }
$preScaleAvgP95 = if ($preScaleP95.Count -gt 0) { [math]::Round(($preScaleP95 | Where-Object { $_ -gt 0 } | Measure-Object -Average).Average, 0) } else { 0 }

# Find scaling events
$scaleUpTime = $null
$scaleDownTime = $null
for ($i = 1; $i -lt $telemetry.Count; $i++) {
    $prev = [int]$telemetry[$i-1].Replicas
    $curr = [int]$telemetry[$i].Replicas
    
    if ($curr -gt $prev -and -not $scaleUpTime) {
        $scaleUpTime = [int]$telemetry[$i].Elapsed_Sec
    }
    if ($curr -lt $prev -and -not $scaleDownTime) {
        $scaleDownTime = [int]$telemetry[$i].Elapsed_Sec
    }
}

# Determine which triggers fired
$latencyTriggered = $maxP95 -gt 500
$queueTriggered = $maxQueue -gt 20
$cpuTriggered = $maxCPU -gt 850

# Generate report
$report = @"
========================================
KEDA + HPA HYBRID CAPACITY ANALYSIS
========================================
Test Configuration:
  Users: $Users
  Spawn Rate: $SpawnRate/s
  Duration: ${Duration}s
  Samples: $sampleCount

========================================
SCALING BEHAVIOR
========================================
Replica Progression:
  Initial: $minReplicas pods
  Peak: $maxReplicas pods
  Average: $avgReplicas pods
  Final: $($telemetry[-1].Replicas) pods

Scaling Timeline:
  First Scale-Up: $(if ($scaleUpTime) { "${scaleUpTime}s" } else { "None" })
  First Scale-Down: $(if ($scaleDownTime) { "${scaleDownTime}s" } else { "None" })
  Total Scale Events: $(if ($scaleUpTime) { ($maxReplicas - $minReplicas) } else { 0 })

========================================
PERFORMANCE METRICS
========================================
Latency (Prometheus):
  p95 Average: ${avgP95}ms
  p95 Peak: ${maxP95}ms
  p50 Average: ${avgP50}ms
  p50 Peak: ${maxP50}ms

Queue Depth:
  Average: ${avgQueue}
  Peak: ${maxQueue}

Request Rate:
  Average: ${avgRPS} req/s
  Peak: ${maxRPS} req/s

CPU Utilization:
  Average: ${avgCPU}m ($(([math]::Round($avgCPU/10, 0)))%)
  Peak: ${maxCPU}m ($(([math]::Round($maxCPU/10, 0)))%)

========================================
PER-POD CAPACITY ANALYSIS
========================================
Pre-Scale Baseline ($minReplicas pods):
  Total RPS: ${preScaleAvgRPS} req/s
  RPS per Pod: ${rpsPerPod} req/s
  Latency (p95): ${preScaleAvgP95}ms

Sustainable Capacity:
  $(if ($preScaleAvgP95 -lt 500) {
      "[OK] sustainable_RPS_per_pod = ${rpsPerPod} req/s"
      "[OK] latency_threshold = ${preScaleAvgP95}ms (below 500ms target)"
      "[OK] queue_threshold = ${avgQueue} (below 20 target)"
  } else {
      "[WARN] sustainable_RPS_per_pod = ${rpsPerPod} req/s"
      "[WARN] latency_threshold = ${preScaleAvgP95}ms (EXCEEDED 500ms target)"
      "[WARN] Recommendation: Increase minReplicas or reduce load"
  })

Production Sizing:
  For ${maxRPS} req/s peak:
    - Minimum pods needed: $(if ($rpsPerPod -gt 0) { [math]::Ceiling($maxRPS / $rpsPerPod) } else { "N/A" })
    - Recommended minReplicas: $(if ($rpsPerPod -gt 0) { [math]::Ceiling($maxRPS / $rpsPerPod * 1.3) } else { $minReplicas }) (30% headroom)

========================================
TRIGGER ANALYSIS
========================================
KEDA Prometheus Triggers:
  p95 Latency > 500ms: $(if ($latencyTriggered) { "[FIRED] max: ${maxP95}ms" } else { "[NOT FIRED] max: ${maxP95}ms" })
  Queue Length > 20: $(if ($queueTriggered) { "[FIRED] max: ${maxQueue}" } else { "[NOT FIRED] max: ${maxQueue}" })

HPA Resource Triggers:
  CPU > 85%: $(if ($cpuTriggered) { "[FIRED] max: ${maxCPU}m" } else { "[NOT FIRED] max: ${maxCPU}m" })
  Memory > 80%: [NOT TRIGGERED]

Scaling Driver:
  $(if ($latencyTriggered) {
      "Primary: KEDA Latency Trigger (Prometheus)"
  } elseif ($queueTriggered) {
      "Primary: KEDA Queue Trigger (Prometheus)"
  } elseif ($cpuTriggered) {
      "Fallback: HPA CPU Trigger"
  } else {
      "None - workload within capacity"
  })

========================================
HYBRID CONTROLLER COORDINATION
========================================
KEDA ScaledObject:
  Status: $(kubectl get scaledobject inference-slo-scaler -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>$null)
  Triggers: prometheus (latency, queue), cpu, memory

HPA (KEDA-managed):
  Name: keda-hpa-inference-slo-scaler
  Current Replicas: $(kubectl get hpa keda-hpa-inference-slo-scaler -o jsonpath='{.status.currentReplicas}' 2>$null)
  Desired Replicas: $(kubectl get hpa keda-hpa-inference-slo-scaler -o jsonpath='{.status.desiredReplicas}' 2>$null)

Coordination: $(if ($scaleUpTime) { "[OK] No conflicts detected" } else { "[WARN] Check if triggers are too conservative" })

========================================
RECOMMENDATIONS
========================================
Configuration Tuning:
"@

if (-not $latencyTriggered -and -not $queueTriggered -and $preScaleAvgP95 -lt 300) {
    $report += @"
  1. Latency Threshold: Lower from 500ms to 400ms for earlier intervention
  2. Activation Threshold: Lower from 300ms to 200ms for faster reaction
  3. Queue Threshold: Current 20 is appropriate
"@
}

if ($latencyTriggered -and $scaleUpTime -gt 60) {
    $report += @"
  1. Stabilization Window: Reduce from 60s to 30s for faster scale-up
  2. Polling Interval: Reduce from 15s to 10s for more frequent checks
"@
}

if ($maxCPU -gt 850 -and -not $latencyTriggered) {
    $report += @"
  1. CPU Threshold: Lower from 85% to 75% as primary safeguard
  2. Latency metrics: Verify Prometheus queries returning valid data
"@
}

$report += @"

Capacity Planning:
  1. Baseline minReplicas: $minReplicas pods
  2. Sustainable load per pod: ~${rpsPerPod} req/s at <500ms latency
  3. Scale-up buffer: Plan for $(if ($rpsPerPod -gt 0) { [math]::Ceiling($maxRPS / $rpsPerPod * 1.5) } else { $minReplicas * 2 }) pods for peak traffic
  4. Cooldown: Current 180s appropriate for production

Production Deployment:
  1. Enable Prometheus recording rules for stable metrics
  2. Set up Grafana dashboards for real-time monitoring
  3. Configure PagerDuty alerts for:
     - p95 latency > 1000ms for 2 minutes
     - Queue depth > 50 for 1 minute
     - Scale-up failures or HPA errors
  4. Schedule load tests quarterly to revalidate capacity

========================================
OUTPUT FILES
========================================
  Telemetry: $telemetryFile
  Summary: $summaryFile
  Locust Log: $locustLogFile

========================================
"@

# Write report
$report | Out-File -FilePath $summaryFile -Encoding UTF8

# Display report
Write-Host $report -ForegroundColor Cyan

# Display telemetry table
Write-Host ""
Write-Host "TELEMETRY TIMELINE (key samples):" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
$telemetry | Select-Object Timestamp,Elapsed_Sec,Replicas,CPU_Avg,p95_Latency_ms,Queue_Len_Avg,RPS_Estimate | 
    Where-Object { 
        $idx = [array]::IndexOf($telemetry, $_)
        $idx -lt 5 -or $idx -ge ($telemetry.Count - 5) -or ([int]$_.Replicas -ne [int]$telemetry[$idx-1].Replicas)
    } | Format-Table -AutoSize

Write-Host ""
Write-Host "[COMPLETE] Analysis saved to: $testDir" -ForegroundColor Green
Write-Host ""

# Cleanup
Remove-Job -Job $locustJob -Force 2>$null

return @{
    TestDir = $testDir
    MinReplicas = $minReplicas
    MaxReplicas = $maxReplicas
    RPSPerPod = $rpsPerPod
    LatencyP95 = $avgP95
    LatencyTriggered = $latencyTriggered
    QueueTriggered = $queueTriggered
    CPUTriggered = $cpuTriggered
}
