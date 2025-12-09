<#
.SYNOPSIS
    Full end-to-end load test with parallel telemetry collection for KEDA Prometheus-based autoscaling.

.DESCRIPTION
    Runs a Locust load test against the inference-http service while simultaneously collecting:
    - Inference pod logs
    - Pod scaling events
    - KEDA ScaledObject and HPA status
    - Prometheus RPS metrics
    
    All collectors run in parallel and are automatically stopped when the load test completes.

.PARAMETER Users
    Number of concurrent Locust users (default: 200)

.PARAMETER SpawnRate
    User spawn rate per second (default: 50)

.PARAMETER Duration
    Test duration in seconds (default: 120)

.PARAMETER OutputDir
    Base output directory (default: load_test_results)

.EXAMPLE
    .\Run-FullLoadTest.ps1 -Users 200 -SpawnRate 50 -Duration 120

.EXAMPLE
    .\Run-FullLoadTest.ps1 -Users 100 -SpawnRate 25 -Duration 60
#>

param(
    [int]$Users = 200,
    [int]$SpawnRate = 50,
    [int]$Duration = 120,
    [string]$OutputDir = "load_test_results"
)

$ErrorActionPreference = "Continue"

# Generate timestamp and create output directory
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
$testDir = Join-Path (Resolve-Path $OutputDir).Path $timestamp
New-Item -ItemType Directory -Path $testDir -Force | Out-Null

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  KEDA Prometheus Autoscaling Load Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nTest Parameters:" -ForegroundColor Yellow
Write-Host "  Users:      $Users"
Write-Host "  Spawn Rate: $SpawnRate/s"
Write-Host "  Duration:   ${Duration}s"
Write-Host "  Output:     $testDir"
Write-Host ""

# Define output files
$locustOutput = Join-Path $testDir "locust_output.txt"
$inferenceLogs = Join-Path $testDir "inference_logs.txt"
$podScaling = Join-Path $testDir "pod_scaling.txt"
$scalerStatus = Join-Path $testDir "scaler.txt"
$rpsMetrics = Join-Path $testDir "rps.txt"

# Track background processes
$processes = @()

try {
    # =========================================================================
    # Start Background Collectors using Start-Process
    # =========================================================================
    
    Write-Host "Starting background collectors..." -ForegroundColor Green
    
    # 1. Inference Pod Logs Collector
    Write-Host "  [1/4] Inference pod logs -> inference_logs.txt"
    $processes += Start-Process -FilePath "kubectl" `
        -ArgumentList "logs", "-l", "app=inference-http", "-f", "--all-containers=true", "--timestamps" `
        -RedirectStandardOutput $inferenceLogs `
        -RedirectStandardError (Join-Path $testDir "inference_logs_err.txt") `
        -NoNewWindow -PassThru
    
    # 2. Pod Scaling Events Collector
    Write-Host "  [2/4] Pod scaling events -> pod_scaling.txt"
    $processes += Start-Process -FilePath "kubectl" `
        -ArgumentList "get", "pods", "-l", "app=inference-http", "-w" `
        -RedirectStandardOutput $podScaling `
        -RedirectStandardError (Join-Path $testDir "pod_scaling_err.txt") `
        -NoNewWindow -PassThru
    
    # 3. KEDA/HPA Status Collector - write header
    Write-Host "  [3/4] KEDA/HPA status -> scaler.txt"
    "=== KEDA/HPA Status Log ===" | Out-File -FilePath $scalerStatus -Encoding utf8
    "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $scalerStatus -Append -Encoding utf8
    "" | Out-File -FilePath $scalerStatus -Append -Encoding utf8
    
    # 4. Prometheus RPS Metrics - write header
    Write-Host "  [4/4] Prometheus RPS metrics -> rps.txt"
    "Timestamp,RPS,P95_Latency_ms,Pod_Count" | Out-File -FilePath $rpsMetrics -Encoding utf8
    
    # Give streaming collectors a moment to start
    Start-Sleep -Seconds 2
    
    # =========================================================================
    # Start Polling Collectors in Background Runspace
    # =========================================================================
    
    # Create a synchronized hashtable for control
    $syncHash = [hashtable]::Synchronized(@{
        StopPolling = $false
        ScalerFile = $scalerStatus
        RpsFile = $rpsMetrics
    })
    
    # Scaler Status Polling Job
    $scalerRunspace = [runspacefactory]::CreateRunspace()
    $scalerRunspace.Open()
    $scalerRunspace.SessionStateProxy.SetVariable("syncHash", $syncHash)
    
    $scalerScript = {
        while (-not $syncHash.StopPolling) {
            $ts = Get-Date -Format "HH:mm:ss"
            $content = @()
            $content += "=== $ts ==="
            $content += ""
            $content += "--- ScaledObject Status ---"
            $content += (kubectl get scaledobject inference-http-rps-scaler -o wide 2>&1)
            $content += ""
            $content += "--- HPA Status ---"
            $content += (kubectl get hpa keda-hpa-inference-http-rps-scaler 2>&1)
            $content += ""
            $content += "--- Pod Count ---"
            $podList = kubectl get pods -l app=inference-http --no-headers 2>&1
            $content += $podList
            $content += ""
            
            $content | Out-File -FilePath $syncHash.ScalerFile -Append -Encoding utf8
            
            Start-Sleep -Seconds 5
        }
    }
    
    $scalerPS = [powershell]::Create().AddScript($scalerScript)
    $scalerPS.Runspace = $scalerRunspace
    $scalerHandle = $scalerPS.BeginInvoke()
    
    # RPS Metrics Polling Job
    $rpsRunspace = [runspacefactory]::CreateRunspace()
    $rpsRunspace.Open()
    $rpsRunspace.SessionStateProxy.SetVariable("syncHash", $syncHash)
    
    $rpsScript = {
        while (-not $syncHash.StopPolling) {
            $ts = Get-Date -Format "HH:mm:ss"
            
            # Query RPS
            $rpsQuery = "sum(rate(inference_jobs_processed_total[1m]))"
            $encodedQuery = [uri]::EscapeDataString($rpsQuery)
            $rpsResult = kubectl exec deployment/prometheus-server -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$encodedQuery" 2>$null
            
            $rps = "N/A"
            if ($rpsResult -match '"value":\[[\d.]+,"([^"]+)"\]') {
                try { $rps = [math]::Round([double]$Matches[1], 2) } catch { $rps = "N/A" }
            }
            
            # Query P95 latency
            $latencyQuery = "histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[2m])) by (le))"
            $encodedLatency = [uri]::EscapeDataString($latencyQuery)
            $latencyResult = kubectl exec deployment/prometheus-server -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$encodedLatency" 2>$null
            
            $p95 = "N/A"
            if ($latencyResult -match '"value":\[[\d.]+,"([^"]+)"\]') {
                try {
                    $p95Val = [double]$Matches[1]
                    if (-not [double]::IsNaN($p95Val) -and -not [double]::IsInfinity($p95Val)) {
                        $p95 = [math]::Round($p95Val * 1000, 2)
                    }
                } catch { $p95 = "N/A" }
            }
            
            # Get pod count
            $podCount = (kubectl get pods -l app=inference-http --no-headers 2>$null | Measure-Object -Line).Lines
            
            "$ts,$rps,$p95,$podCount" | Out-File -FilePath $syncHash.RpsFile -Append -Encoding utf8
            
            Start-Sleep -Seconds 5
        }
    }
    
    $rpsPS = [powershell]::Create().AddScript($rpsScript)
    $rpsPS.Runspace = $rpsRunspace
    $rpsHandle = $rpsPS.BeginInvoke()
    
    # =========================================================================
    # Run Locust Load Test (Foreground)
    # =========================================================================
    
    Write-Host "`nStarting Locust load test..." -ForegroundColor Green
    Write-Host "  Host: http://inference-http:8000"
    Write-Host "  Duration: ${Duration}s"
    Write-Host ""
    
    # Get locust master pod
    $locustMaster = kubectl get pods -l app=locust,role=master -o jsonpath='{.items[0].metadata.name}' 2>$null
    if (-not $locustMaster) {
        throw "Could not find locust-master pod. Is Locust deployed?"
    }
    
    Write-Host "Using Locust master: $locustMaster" -ForegroundColor Gray
    Write-Host ""
    Write-Host "--- Locust Output ---" -ForegroundColor Yellow
    
    # Run Locust and capture output
    $locustArgs = @(
        "exec", $locustMaster, "--",
        "locust", "--headless",
        "--host=http://inference-http:8000",
        "-u", $Users,
        "-r", $SpawnRate,
        "-t", "${Duration}s",
        "--print-stats"
    )
    
    $output = & kubectl @locustArgs 2>&1
    $output | Tee-Object -FilePath $locustOutput
    
    Write-Host ""
    Write-Host "--- End Locust Output ---" -ForegroundColor Yellow
    
} finally {
    # =========================================================================
    # Stop All Background Collectors
    # =========================================================================
    
    Write-Host "`nStopping background collectors..." -ForegroundColor Yellow
    
    # Signal polling jobs to stop
    $syncHash.StopPolling = $true
    Start-Sleep -Seconds 2
    
    # Clean up runspaces
    if ($scalerPS) {
        $scalerPS.Stop()
        $scalerPS.Dispose()
        $scalerRunspace.Close()
    }
    if ($rpsPS) {
        $rpsPS.Stop()
        $rpsPS.Dispose()
        $rpsRunspace.Close()
    }
    
    # Stop streaming processes
    foreach ($proc in $processes) {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    
    Write-Host "All collectors stopped." -ForegroundColor Green
}

# =========================================================================
# Generate Summary
# =========================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Load Test Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nOutput files saved to: $testDir" -ForegroundColor Green
Write-Host ""

# List files with sizes
Get-ChildItem $testDir -File | Where-Object { $_.Length -gt 0 } | ForEach-Object {
    $size = if ($_.Length -gt 1KB) { "{0:N1} KB" -f ($_.Length / 1KB) } else { "$($_.Length) B" }
    Write-Host ("  {0,-25} {1,10}" -f $_.Name, $size)
}

# Quick summary from RPS file
Write-Host "`n--- Quick Metrics Summary ---" -ForegroundColor Yellow
if (Test-Path $rpsMetrics) {
    $rpsData = Import-Csv $rpsMetrics -ErrorAction SilentlyContinue
    if ($rpsData -and $rpsData.Count -gt 0) {
        $validRps = $rpsData | Where-Object { $_.RPS -ne "N/A" -and $_.RPS -ne "" }
        if ($validRps) {
            $maxRps = ($validRps | ForEach-Object { [double]$_.RPS } | Measure-Object -Maximum).Maximum
            $avgRps = ($validRps | ForEach-Object { [double]$_.RPS } | Measure-Object -Average).Average
            Write-Host "  Max RPS:     $([math]::Round($maxRps, 2))"
            Write-Host "  Avg RPS:     $([math]::Round($avgRps, 2))"
        }
        $maxPods = ($rpsData | ForEach-Object { [int]$_.Pod_Count } | Measure-Object -Maximum).Maximum
        $minPods = ($rpsData | ForEach-Object { [int]$_.Pod_Count } | Measure-Object -Minimum).Minimum
        Write-Host "  Pod Range:   $minPods -> $maxPods"
    }
}

# Extract scaling events from scaler log
if (Test-Path $scalerStatus) {
    $content = Get-Content $scalerStatus -Raw
    $scaleChanges = [regex]::Matches($content, 'inference-http-rps-scaler\s+\S+\s+\S+\s+(\d+)\s+(\d+)\s+(\w+)\s+(\w+)')
    if ($scaleChanges.Count -gt 0) {
        Write-Host "`n--- Scaling Events ---" -ForegroundColor Yellow
        $lastReplicas = 0
        foreach ($match in $scaleChanges) {
            $replicas = [int]$match.Groups[1].Value
            if ($replicas -ne $lastReplicas) {
                Write-Host "  Replicas: $lastReplicas -> $replicas"
                $lastReplicas = $replicas
            }
        }
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Test Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan
