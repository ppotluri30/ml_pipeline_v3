# KEDA HTTP Load Test with Real-Time Monitoring
# This script runs a Locust load test through KEDA proxy while monitoring scaling

param(
    [int]$Users = 200,
    [int]$RampUp = 50,
    [int]$Duration = 120,
    [string]$OutputDir = "c:\Users\ppotluri\Desktop\ml_pipeline_v3\test_results"
)

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$testDir = "$OutputDir\keda_test_$timestamp"
New-Item -ItemType Directory -Path $testDir -Force | Out-Null

Write-Host "=== KEDA HTTP Load Test with Monitoring ===" -ForegroundColor Cyan
Write-Host "Test ID: $timestamp" -ForegroundColor Yellow
Write-Host "Users: $Users, Ramp-up: $RampUp/s, Duration: $Duration`s" -ForegroundColor Yellow
Write-Host "Output: $testDir" -ForegroundColor Yellow
Write-Host ""

# CSV files
$podLogFile = "$testDir\pod_scaling.csv"
$queueLogFile = "$testDir\queue_metrics.csv"
$scalerLogFile = "$testDir\scaler_events.txt"
$testLogFile = "$testDir\load_test.txt"

# Initialize CSV headers
"Timestamp,Elapsed_s,Pod_Count,Pod_Names" | Out-File -FilePath $podLogFile -Encoding utf8
"Timestamp,Elapsed_s,Pod_Count,RPS,Concurrency" | Out-File -FilePath $queueLogFile -Encoding utf8

# Function to monitor pods
$monitorPodsScript = {
    param($logFile, $duration)
    $startTime = Get-Date
    $i = 0
    while (((Get-Date) - $startTime).TotalSeconds -lt ($duration + 30)) {
        try {
            $elapsed = [math]::Round(((Get-Date) - $startTime).TotalSeconds)
            $ts = (Get-Date).ToString("HH:mm:ss")
            $pods = kubectl get pods -l app=inference-http --no-headers 2>$null
            if ($pods) {
                $podCount = ($pods | Measure-Object).Count
                $podNames = ($pods | ForEach-Object { ($_ -split '\s+')[0] }) -join ';'
                "$ts,$elapsed,$podCount,$podNames" | Out-File -Append -FilePath $logFile -Encoding utf8
                Write-Host "[$ts] T+$($elapsed)s | Pods: $podCount" -ForegroundColor Green
            }
        } catch {
            Write-Host "Pod monitoring error: $_" -ForegroundColor Red
        }
        Start-Sleep -Seconds 3
        $i++
    }
}

# Function to monitor queue
$monitorQueueScript = {
    param($logFile, $duration)
    $startTime = Get-Date
    $i = 0
    while (((Get-Date) - $startTime).TotalSeconds -lt ($duration + 30)) {
        try {
            $elapsed = [math]::Round(((Get-Date) - $startTime).TotalSeconds)
            $ts = (Get-Date).ToString("HH:mm:ss")
            $pods = (kubectl get pods -l app=inference-http --no-headers 2>$null | Measure-Object).Count
            
            # Query queue metrics
            $queueJson = kubectl run curl-mon-$i --rm -it --restart=Never --image=curlimages/curl:8.10.1 -- curl -s http://keda-add-ons-http-interceptor-admin.keda.svc.cluster.local:9090/queue 2>$null
            
            $rps = 0
            $conc = 0
            if ($queueJson -match '"RPS":(\d+\.?\d*)') { $rps = [math]::Round([double]$Matches[1], 2) }
            if ($queueJson -match '"Concurrency":(\d+\.?\d*)') { $conc = [math]::Round([double]$Matches[1], 2) }
            
            "$ts,$elapsed,$pods,$rps,$conc" | Out-File -Append -FilePath $logFile -Encoding utf8
            Write-Host "[$ts] T+$($elapsed)s | RPS: $rps | Concurrency: $conc" -ForegroundColor Cyan
        } catch {
            Write-Host "Queue monitoring error: $_" -ForegroundColor Red
        }
        Start-Sleep -Seconds 5
        $i++
    }
}

# Check initial state
Write-Host "`n[Pre-Test] Checking initial state..." -ForegroundColor Yellow
$initialPods = kubectl get pods -l app=inference-http --no-headers
Write-Host "Initial pods:" -ForegroundColor Gray
Write-Host $initialPods

$initialQueue = kubectl run curl-init --rm -it --restart=Never --image=curlimages/curl:8.10.1 -- curl -s http://keda-add-ons-http-interceptor-admin.keda.svc.cluster.local:9090/queue 2>$null
Write-Host "Initial queue: $initialQueue" -ForegroundColor Gray

# Get Locust master pod
$master = kubectl get pods -l app=locust,role=master -o jsonpath='{.items[0].metadata.name}' 2>$null
if (-not $master) {
    Write-Host "ERROR: Locust master pod not found!" -ForegroundColor Red
    exit 1
}
Write-Host "`nLocust master: $master" -ForegroundColor Green

# Start monitoring jobs
Write-Host "`n[Starting Monitors]" -ForegroundColor Yellow
$podMonitorJob = Start-Job -ScriptBlock $monitorPodsScript -ArgumentList $podLogFile, $Duration
Write-Host "Pod monitor started (Job ID: $($podMonitorJob.Id))" -ForegroundColor Gray

$queueMonitorJob = Start-Job -ScriptBlock $monitorQueueScript -ArgumentList $queueLogFile, $Duration
Write-Host "Queue monitor started (Job ID: $($queueMonitorJob.Id))" -ForegroundColor Gray

Start-Sleep -Seconds 2

# Run load test
Write-Host "`n[Starting Load Test]" -ForegroundColor Yellow
Write-Host "Command: locust --headless --host=http://inference-http:8000 -u $Users -r $RampUp -t $($Duration)s" -ForegroundColor Gray

$loadTestStartTime = Get-Date
try {
    kubectl exec $master -- locust `
        --headless `
        --host http://keda-add-ons-http-interceptor-proxy.keda.svc.cluster.local:8080 `
        -H "Host: inference-http" `
        -u $Users `
        -r $RampUp `
        -t "$($Duration)s" `
        --print-stats 2>&1 | Tee-Object -FilePath $testLogFile
} catch {
    Write-Host "Load test error: $_" -ForegroundColor Red
}
$loadTestEndTime = Get-Date
$actualDuration = ($loadTestEndTime - $loadTestStartTime).TotalSeconds

Write-Host "`n[Load Test Complete]" -ForegroundColor Green
Write-Host "Actual duration: $([math]::Round($actualDuration, 1))s" -ForegroundColor Gray

# Wait for monitors to finish
Write-Host "`n[Waiting for monitors to complete...]" -ForegroundColor Yellow
Start-Sleep -Seconds 10

Stop-Job -Job $podMonitorJob -ErrorAction SilentlyContinue
Stop-Job -Job $queueMonitorJob -ErrorAction SilentlyContinue
Remove-Job -Job $podMonitorJob -Force -ErrorAction SilentlyContinue
Remove-Job -Job $queueMonitorJob -Force -ErrorAction SilentlyContinue

# Post-test state
Write-Host "`n[Post-Test] Checking final state..." -ForegroundColor Yellow
$finalPods = kubectl get pods -l app=inference-http --no-headers
Write-Host "Final pods:" -ForegroundColor Gray
Write-Host $finalPods

$finalQueue = kubectl run curl-final --rm -it --restart=Never --image=curlimages/curl:8.10.1 -- curl -s http://keda-add-ons-http-interceptor-admin.keda.svc.cluster.local:9090/queue 2>$null
Write-Host "Final queue: $finalQueue" -ForegroundColor Gray

# Parse results
Write-Host "`n=== TEST SUMMARY ===" -ForegroundColor Cyan
Write-Host "Test ID: $timestamp" -ForegroundColor Yellow
Write-Host "Results directory: $testDir" -ForegroundColor Yellow

if (Test-Path $testLogFile) {
    $logContent = Get-Content $testLogFile -Raw
    
    # Extract stats
    if ($logContent -match 'Total:\s+(\d+)\s+Success:\s+(\d+)\s+Fail:\s+(\d+)') {
        $total = $Matches[1]
        $success = $Matches[2]
        $fail = $Matches[3]
        $successRate = [math]::Round(100 * [double]$success / [double]$total, 1)
        
        Write-Host "`nLoad Test Results:" -ForegroundColor Green
        Write-Host "  Total Requests: $total"
        Write-Host "  Successes: $success"
        Write-Host "  Failures: $fail"
        Write-Host "  Success Rate: $successRate%"
    }
    
    if ($logContent -match 'P50:\s+(\d+)\s+ms.*P95:\s+(\d+)\s+ms.*P99:\s+(\d+)\s+ms') {
        Write-Host "`nLatency:" -ForegroundColor Green
        Write-Host "  P50: $($Matches[1]) ms"
        Write-Host "  P95: $($Matches[2]) ms"
        Write-Host "  P99: $($Matches[3]) ms"
    }
    
    if ($logContent -match 'RPS:\s+([\d.]+)') {
        Write-Host "  Throughput: $($Matches[1]) RPS"
    }
}

# Scaling summary
if (Test-Path $podLogFile) {
    $podData = Import-Csv $podLogFile
    $minPods = ($podData.Pod_Count | Measure-Object -Minimum).Minimum
    $maxPods = ($podData.Pod_Count | Measure-Object -Maximum).Maximum
    
    Write-Host "`nScaling Behavior:" -ForegroundColor Green
    Write-Host "  Min Pods: $minPods"
    Write-Host "  Max Pods: $maxPods"
    Write-Host "  Scaling Range: $minPods -> $maxPods"
}

Write-Host "`nFiles generated:" -ForegroundColor Yellow
Write-Host "  Pod scaling: $podLogFile"
Write-Host "  Queue metrics: $queueLogFile"
Write-Host "  Load test log: $testLogFile"

Write-Host "`n=== TEST COMPLETE ===" -ForegroundColor Cyan
