# ============================================================================
# HPA Performance Testing Suite
# ============================================================================
# Automated horizontal scaling performance tests for inference service
# Tests multiple concurrency levels and monitors HPA autoscaling behavior
# ============================================================================

param(
    [string]$OutputDir = ".\reports\hpa_performance",
    [string]$ResultsFile = "HPA_PERFORMANCE_RESULTS.csv",
    [string]$SummaryFile = "HPA_PERFORMANCE_SUMMARY.txt",
    [string]$InferenceDeployment = "inference",
    [string]$LocustMaster = "deployment/locust-master",
    [string]$TargetHost = "http://inference:8000"
)

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$ResultsCsvPath = Join-Path $OutputDir $ResultsFile
$SummaryPath = Join-Path $OutputDir $SummaryFile

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  HPA PERFORMANCE TESTING SUITE" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Test Matrix Definition
# ============================================================================
$testMatrix = @(
    # Light load - baseline
    @{Users=10; SpawnRate=2; Duration=30; Description="Light load baseline"},
    @{Users=10; SpawnRate=2; Duration=60; Description="Light load sustained"},
    
    # Medium load - trigger HPA
    @{Users=25; SpawnRate=5; Duration=30; Description="Medium load quick"},
    @{Users=25; SpawnRate=5; Duration=60; Description="Medium load sustained"},
    @{Users=25; SpawnRate=5; Duration=120; Description="Medium load extended"},
    
    # Heavy load - stress HPA
    @{Users=50; SpawnRate=10; Duration=30; Description="Heavy load quick"},
    @{Users=50; SpawnRate=10; Duration=60; Description="Heavy load sustained"},
    @{Users=50; SpawnRate=10; Duration=120; Description="Heavy load extended"},
    
    # Extreme load - max capacity
    @{Users=100; SpawnRate=10; Duration=30; Description="Extreme load quick"},
    @{Users=100; SpawnRate=10; Duration=60; Description="Extreme load sustained"},
    @{Users=100; SpawnRate=10; Duration=120; Description="Extreme load extended"}
)

Write-Host "📋 Test Matrix: $($testMatrix.Count) scenarios" -ForegroundColor Yellow
Write-Host ""

# Initialize results array
$results = @()

# ============================================================================
# Helper Functions
# ============================================================================

function Get-HPAMetrics {
    param([string]$Deployment)
    
    $hpaJson = kubectl get hpa $Deployment -o json 2>$null | ConvertFrom-Json
    
    if ($hpaJson) {
        $currentReplicas = $hpaJson.status.currentReplicas
        $desiredReplicas = $hpaJson.status.desiredReplicas
        $currentCPU = 0
        
        # Extract CPU utilization
        if ($hpaJson.status.currentMetrics) {
            foreach ($metric in $hpaJson.status.currentMetrics) {
                if ($metric.type -eq "Resource" -and $metric.resource.name -eq "cpu") {
                    $currentCPU = $metric.resource.current.averageUtilization
                    break
                }
            }
        }
        
        return @{
            CurrentReplicas = $currentReplicas
            DesiredReplicas = $desiredReplicas
            CPUUtilization = $currentCPU
            Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        }
    }
    
    return $null
}

function Get-PodCount {
    param([string]$Deployment)
    
    $pods = kubectl get pods -l app=$Deployment -o json | ConvertFrom-Json
    $running = ($pods.items | Where-Object { $_.status.phase -eq "Running" }).Count
    $total = $pods.items.Count
    
    return @{
        Running = $running
        Total = $total
    }
}

function Parse-LocustOutput {
    param([string]$Output)
    
    # Parse the summary table
    $lines = $Output -split "`n"
    
    $totalRequests = 0
    $totalFailures = 0
    $avgResponseTime = 0
    $p95ResponseTime = 0
    $throughput = 0
    
    foreach ($line in $lines) {
        # Look for the POST predict line - use simpler pattern matching
        if ($line -match 'POST\s+predict\s+(\d+)\s+(\d+)') {
            $totalRequests = [int]$Matches[1]
            $totalFailures = [int]$Matches[2]
        }
        
        # Extract avg response time from the same line
        if ($line -match 'POST\s+predict.*\|\s+(\d+\.?\d*)\s+\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*\s+\|\s+(\d+\.?\d*)') {
            $avgResponseTime = [decimal]$Matches[1]
            $throughput = [decimal]$Matches[2]
        }
        
        # Look for p95 in percentiles table
        if ($line -match 'POST\s+predict\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+\.?\d*)') {
            $p95ResponseTime = [decimal]$Matches[1]
        }
    }
    
    return @{
        TotalRequests = $totalRequests
        TotalFailures = $totalFailures
        FailureRate = if ($totalRequests -gt 0) { ($totalFailures / $totalRequests) * 100 } else { 0 }
        AvgResponseTime = $avgResponseTime
        P95ResponseTime = $p95ResponseTime
        Throughput = $throughput
    }
}

function Run-LocustTest {
    param(
        [int]$Users,
        [int]$SpawnRate,
        [int]$Duration,
        [string]$TargetHost
    )
    
    $usersTxt = $Users
    $spawnTxt = $SpawnRate
    $durTxt = $Duration
    Write-Host "  Running Locust: $usersTxt users at ${spawnTxt}/s for ${durTxt}s..." -ForegroundColor Cyan
    
    $command = "cd /home/locust; locust -f locustfile.py --headless --host=$TargetHost -u $Users -r $SpawnRate -t ${Duration}s --only-summary"
    
    $output = kubectl exec $LocustMaster -- sh -c $command 2>&1 | Out-String
    
    return $output
}

function Monitor-HPADuringTest {
    param(
        [int]$DurationSeconds,
        [string]$Deployment
    )
    
    $snapshots = @()
    $interval = 10  # Poll every 10 seconds
    $iterations = [Math]::Ceiling($DurationSeconds / $interval)
    
    for ($i = 0; $i -lt $iterations; $i++) {
        Start-Sleep -Seconds $interval
        $metrics = Get-HPAMetrics -Deployment $Deployment
        if ($metrics) {
            $snapshots += $metrics
            Write-Host "    📊 HPA: Replicas=$($metrics.CurrentReplicas), CPU=$($metrics.CPUUtilization)%" -ForegroundColor DarkGray
        }
    }
    
    return $snapshots
}

function Calculate-HPAAction {
    param([array]$Snapshots)
    
    if ($Snapshots.Count -lt 2) {
        return "insufficient_data"
    }
    
    $initialReplicas = $Snapshots[0].CurrentReplicas
    $finalReplicas = $Snapshots[-1].CurrentReplicas
    
    $maxReplicas = ($Snapshots | Measure-Object -Property CurrentReplicas -Maximum).Maximum
    
    if ($maxReplicas -gt $initialReplicas) {
        return "scaled_up"
    } elseif ($finalReplicas -lt $initialReplicas) {
        return "scaled_down"
    } else {
        return "steady"
    }
}

function Get-AverageCPU {
    param([array]$Snapshots)
    
    if ($Snapshots.Count -eq 0) {
        return 0
    }
    
    $avgCPU = ($Snapshots | Measure-Object -Property CPUUtilization -Average).Average
    return [Math]::Round($avgCPU, 1)
}

function Get-MaxReplicas {
    param([array]$Snapshots)
    
    if ($Snapshots.Count -eq 0) {
        return 0
    }
    
    return ($Snapshots | Measure-Object -Property CurrentReplicas -Maximum).Maximum
}

# ============================================================================
# Pre-Test Validation
# ============================================================================

Write-Host "🔍 Pre-test validation..." -ForegroundColor Yellow

# Check HPA exists
$hpaExists = kubectl get hpa $InferenceDeployment 2>$null
if (-not $hpaExists) {
    Write-Host "❌ HPA '$InferenceDeployment' not found!" -ForegroundColor Red
    Write-Host "   Creating basic HPA..." -ForegroundColor Yellow
    kubectl autoscale deployment $InferenceDeployment --cpu-percent=70 --min=2 --max=10
    Start-Sleep -Seconds 5
}

# Check Locust master
$locustPod = kubectl get $LocustMaster 2>$null
if (-not $locustPod) {
    Write-Host "❌ Locust master not found at: $LocustMaster" -ForegroundColor Red
    exit 1
}

# Get initial state
$initialHPA = Get-HPAMetrics -Deployment $InferenceDeployment
$initialPods = Get-PodCount -Deployment $InferenceDeployment

Write-Host "✅ Initial state:" -ForegroundColor Green
Write-Host "   HPA: $($initialHPA.CurrentReplicas) replicas, $($initialHPA.CPUUtilization)% CPU"
Write-Host "   Pods: $($initialPods.Running)/$($initialPods.Total) running"
Write-Host ""

# ============================================================================
# Initialize CSV
# ============================================================================

$csvHeaders = "TestID,Users,SpawnRate,Duration,Description,TotalRequests,Failures,FailureRate,AvgLatency,P95Latency,Throughput,InitialReplicas,MaxReplicas,FinalReplicas,AvgCPU,HPAAction,Timestamp"
$csvHeaders | Out-File -FilePath $ResultsCsvPath -Encoding UTF8

Write-Host "📝 Results will be saved to: $ResultsCsvPath" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Execute Test Suite
# ============================================================================

$testNumber = 1
$totalTests = $testMatrix.Count

foreach ($test in $testMatrix) {
    $testDesc = $test.Description
    $testTotal = $totalTests
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host "TEST $testNumber of $testTotal - $testDesc" -ForegroundColor Cyan
    Write-Host "  Users: $($test.Users) | Spawn: $($test.SpawnRate)/s | Duration: $($test.Duration)s" -ForegroundColor Cyan
    Write-Host "============================================================================" -ForegroundColor Cyan
    
    # Get pre-test state
    $preTestHPA = Get-HPAMetrics -Deployment $InferenceDeployment
    $preTestPods = Get-PodCount -Deployment $InferenceDeployment
    
    Write-Host "  📊 Pre-test: $($preTestHPA.CurrentReplicas) replicas, $($preTestHPA.CPUUtilization)% CPU" -ForegroundColor Yellow
    
    # Start monitoring in background
    $monitorJob = Start-Job -ScriptBlock {
        param($Duration, $Deployment)
        
        $snapshots = @()
        $interval = 10
        $iterations = [Math]::Ceiling($Duration / $interval)
        
        for ($i = 0; $i -lt $iterations; $i++) {
            Start-Sleep -Seconds $interval
            
            $hpaJson = kubectl get hpa $Deployment -o json 2>$null | ConvertFrom-Json
            if ($hpaJson) {
                $currentReplicas = $hpaJson.status.currentReplicas
                $desiredReplicas = $hpaJson.status.desiredReplicas
                $currentCPU = 0
                
                if ($hpaJson.status.currentMetrics) {
                    foreach ($metric in $hpaJson.status.currentMetrics) {
                        if ($metric.type -eq "Resource" -and $metric.resource.name -eq "cpu") {
                            $currentCPU = $metric.resource.current.averageUtilization
                            break
                        }
                    }
                }
                
                $snapshots += @{
                    CurrentReplicas = $currentReplicas
                    DesiredReplicas = $desiredReplicas
                    CPUUtilization = $currentCPU
                    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                }
            }
        }
        
        return $snapshots
    } -ArgumentList $test.Duration, $InferenceDeployment
    
    # Run Locust test
    $locustOutput = Run-LocustTest -Users $test.Users -SpawnRate $test.SpawnRate -Duration $test.Duration -TargetHost $TargetHost
    
    # Wait for monitoring to complete
    $hpaSnapshots = Receive-Job -Job $monitorJob -Wait
    Remove-Job -Job $monitorJob
    
    # Get post-test state
    Start-Sleep -Seconds 5  # Allow metrics to stabilize
    $postTestHPA = Get-HPAMetrics -Deployment $InferenceDeployment
    $postTestPods = Get-PodCount -Deployment $InferenceDeployment
    
    # Parse Locust results
    $locustMetrics = Parse-LocustOutput -Output $locustOutput
    
    # Calculate HPA metrics
    $hpaAction = Calculate-HPAAction -Snapshots $hpaSnapshots
    $avgCPU = Get-AverageCPU -Snapshots $hpaSnapshots
    $maxReplicas = Get-MaxReplicas -Snapshots $hpaSnapshots
    
    # Display results
    Write-Host ""
    Write-Host "  Test Complete!" -ForegroundColor Green
    $failPct = [Math]::Round($locustMetrics.FailureRate, 2)
    Write-Host "     Requests: $($locustMetrics.TotalRequests) | Failures: $($locustMetrics.TotalFailures) ($failPct percent)" -ForegroundColor White
    Write-Host "     Throughput: $([Math]::Round($locustMetrics.Throughput, 2)) req/s" -ForegroundColor White
    Write-Host "     Latency: Avg=$([Math]::Round($locustMetrics.AvgResponseTime, 0))ms, P95=$([Math]::Round($locustMetrics.P95ResponseTime, 0))ms" -ForegroundColor White
    Write-Host "     Replicas: $($preTestHPA.CurrentReplicas) → $maxReplicas (max) → $($postTestHPA.CurrentReplicas)" -ForegroundColor White
    Write-Host "     CPU: $avgCPU% (avg during test)" -ForegroundColor White
    Write-Host "     HPA Action: $hpaAction" -ForegroundColor $(if ($hpaAction -eq "scaled_up") { "Yellow" } elseif ($hpaAction -eq "scaled_down") { "Cyan" } else { "White" })
    Write-Host ""
    
    # Save to CSV
    $csvLine = "$testNumber,$($test.Users),$($test.SpawnRate),$($test.Duration),`"$($test.Description)`",$($locustMetrics.TotalRequests),$($locustMetrics.TotalFailures),$([Math]::Round($locustMetrics.FailureRate, 2)),$([Math]::Round($locustMetrics.AvgResponseTime, 1)),$([Math]::Round($locustMetrics.P95ResponseTime, 1)),$([Math]::Round($locustMetrics.Throughput, 2)),$($preTestHPA.CurrentReplicas),$maxReplicas,$($postTestHPA.CurrentReplicas),$avgCPU,$hpaAction,$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $csvLine | Out-File -FilePath $ResultsCsvPath -Append -Encoding UTF8
    
    # Store for summary
    $results += @{
        TestID = $testNumber
        Users = $test.Users
        SpawnRate = $test.SpawnRate
        Duration = $test.Duration
        Description = $test.Description
        TotalRequests = $locustMetrics.TotalRequests
        Failures = $locustMetrics.TotalFailures
        FailureRate = $locustMetrics.FailureRate
        AvgLatency = $locustMetrics.AvgResponseTime
        P95Latency = $locustMetrics.P95ResponseTime
        Throughput = $locustMetrics.Throughput
        InitialReplicas = $preTestHPA.CurrentReplicas
        MaxReplicas = $maxReplicas
        FinalReplicas = $postTestHPA.CurrentReplicas
        AvgCPU = $avgCPU
        HPAAction = $hpaAction
    }
    
    $testNumber++
    
    # Cool-down period between tests
    if ($testNumber -le $totalTests) {
        Write-Host "  ⏳ Cool-down: 30 seconds..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 30
    }
}

# ============================================================================
# Generate Summary Report
# ============================================================================

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  PERFORMANCE SUITE SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Console table
$tableFormat = "{0,-6} {1,-7} {2,-7} {3,-9} {4,-12} {5,-8} {6,-8} {7,-10}"
Write-Host ($tableFormat -f "USERS", "SPAWN", "TIME(s)", "REPLICAS", "THROUGHPUT", "P95(ms)", "CPU(%)", "HPA_ACTION") -ForegroundColor Yellow

foreach ($result in $results) {
    $replicasDisplay = "$($result.InitialReplicas)→$($result.MaxReplicas)"
    $throughputDisplay = [Math]::Round($result.Throughput, 1)
    $p95Display = [Math]::Round($result.P95Latency, 0)
    
    Write-Host ($tableFormat -f $result.Users, $result.SpawnRate, $result.Duration, $replicasDisplay, $throughputDisplay, $p95Display, $result.AvgCPU, $result.HPAAction)
}

# Validation summary
Write-Host ""
Write-Host "🎯 VALIDATION RESULTS:" -ForegroundColor Yellow

$totalFailures = ($results | Measure-Object -Property Failures -Sum).Sum
$maxP95 = ($results | Measure-Object -Property P95Latency -Maximum).Maximum
$scaledUpTests = ($results | Where-Object { $_.HPAAction -eq "scaled_up" }).Count

$allPassed = $true

Write-Host "   Total Failures: $totalFailures" -ForegroundColor $(if ($totalFailures -eq 0) { "Green" } else { "Red" })
if ($totalFailures -eq 0) {
    Write-Host "   ✅ No request failures across all scenarios" -ForegroundColor Green
} else {
    Write-Host "   ❌ Request failures detected" -ForegroundColor Red
    $allPassed = $false
}

Write-Host "   Max P95 Latency: $([Math]::Round($maxP95, 0))ms" -ForegroundColor $(if ($maxP95 -lt 1000) { "Green" } else { "Red" })
if ($maxP95 -lt 1000) {
    Write-Host "   ✅ P95 latency < 1s across all scenarios" -ForegroundColor Green
} else {
    Write-Host "   ❌ P95 latency exceeded 1s threshold" -ForegroundColor Red
    $allPassed = $false
}

Write-Host "   HPA Scale-up Events: $scaledUpTests / $($results.Count) tests" -ForegroundColor Cyan
if ($scaledUpTests -gt 0) {
    Write-Host "   ✅ HPA responded to increased load" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  HPA did not scale up (may need tuning)" -ForegroundColor Yellow
}

Write-Host ""
if ($allPassed) {
    Write-Host "🎉 ALL VALIDATION CRITERIA PASSED!" -ForegroundColor Green
} else {
    Write-Host "⚠️  SOME VALIDATION CRITERIA FAILED" -ForegroundColor Yellow
}

# Save detailed summary to file
$summaryContent = @"
============================================================================
HPA PERFORMANCE TESTING SUITE - SUMMARY REPORT
============================================================================
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Total Tests: $($results.Count)

VALIDATION RESULTS:
- Total Request Failures: $totalFailures (Target: 0)
- Max P95 Latency: $([Math]::Round($maxP95, 0))ms (Target: < 1000ms)
- HPA Scale-up Events: $scaledUpTests / $($results.Count) tests

DETAILED RESULTS:

USERS  SPAWN  TIME(s)  REPLICAS    THROUGHPUT  P95(ms)  CPU(%)  HPA_ACTION
"@

foreach ($result in $results) {
    $line = "{0,-6} {1,-7} {2,-9} {3,-12} {4,-12} {5,-8} {6,-8} {7}" -f `
        $result.Users, `
        $result.SpawnRate, `
        $result.Duration, `
        "$($result.InitialReplicas)→$($result.MaxReplicas)", `
        [Math]::Round($result.Throughput, 1), `
        [Math]::Round($result.P95Latency, 0), `
        $result.AvgCPU, `
        $result.HPAAction
    
    $summaryContent += "`n$line"
}

$summaryContent += @"


KEY INSIGHTS:

Load Threshold for HPA Scaling:
"@

$scalingTests = $results | Where-Object { $_.HPAAction -eq 'scaled_up' } | Sort-Object -Property Users
if ($scalingTests.Count -gt 0) {
    $firstScaling = $scalingTests[0]
    $summaryContent += "`n- HPA first scaled at $($firstScaling.Users) users with $($firstScaling.AvgCPU)% CPU"
} else {
    $summaryContent += "`n- HPA did not scale during any test (consider increasing load or lowering threshold)"
}

$highCPUTests = $results | Where-Object { $_.AvgCPU -gt 70 } | Sort-Object -Property AvgCPU -Descending
if ($highCPUTests.Count -gt 0) {
    $summaryContent += "`n- $($highCPUTests.Count) tests exceeded 70% CPU utilization"
    $summaryContent += "`n- Max CPU: $($highCPUTests[0].AvgCPU)% at $($highCPUTests[0].Users) users"
}

$summaryContent += @"


Performance Trends:
- Throughput range: $([Math]::Round(($results | Measure-Object -Property Throughput -Minimum).Minimum, 1)) - $([Math]::Round(($results | Measure-Object -Property Throughput -Maximum).Maximum, 1)) req/s
- P95 Latency range: $([Math]::Round(($results | Measure-Object -Property P95Latency -Minimum).Minimum, 0)) - $([Math]::Round(($results | Measure-Object -Property P95Latency -Maximum).Maximum, 0))ms
- Replica range: $($results[0].InitialReplicas) - $(($results | Measure-Object -Property MaxReplicas -Maximum).Maximum)

============================================================================
Raw data available in: $ResultsCsvPath
============================================================================
"@

$summaryContent | Out-File -FilePath $SummaryPath -Encoding UTF8

Write-Host ""
Write-Host "📄 Detailed summary saved to: $SummaryPath" -ForegroundColor Green
Write-Host "📊 Raw CSV data saved to: $ResultsCsvPath" -ForegroundColor Green
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  TEST SUITE COMPLETE!" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
