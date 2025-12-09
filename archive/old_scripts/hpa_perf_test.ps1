# HPA Performance Testing Suite
# Tests inference service autoscaling across multiple load scenarios

param(
    [string]$OutputDir = ".\reports\hpa_performance"
)

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$ResultsFile = Join-Path $OutputDir "HPA_PERFORMANCE_RESULTS.csv"
$SummaryFile = Join-Path $OutputDir "HPA_PERFORMANCE_SUMMARY.txt"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "HPA PERFORMANCE TESTING SUITE" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Test matrix
$tests = @(
    @{Users=10; Spawn=2; Time=30; Desc="Light baseline"},
    @{Users=10; Spawn=2; Time=60; Desc="Light sustained"},
    @{Users=25; Spawn=5; Time=30; Desc="Medium quick"},
    @{Users=25; Spawn=5; Time=60; Desc="Medium sustained"},
    @{Users=25; Spawn=5; Time=120; Desc="Medium extended"},
    @{Users=50; Spawn=10; Time=30; Desc="Heavy quick"},
    @{Users=50; Spawn=10; Time=60; Desc="Heavy sustained"},
    @{Users=50; Spawn=10; Time=120; Desc="Heavy extended"},
    @{Users=100; Spawn=10; Time=30; Desc="Extreme quick"},
    @{Users=100; Spawn=10; Time=60; Desc="Extreme sustained"},
    @{Users=100; Spawn=10; Time=120; Desc="Extreme extended"}
)

Write-Host "Test Matrix: $($tests.Count) scenarios" -ForegroundColor Yellow
Write-Host ""

# Initialize CSV
"TestID,Users,SpawnRate,Duration,Description,TotalRequests,Failures,AvgLatency,P95Latency,Throughput,InitialReplicas,MaxReplicas,FinalReplicas,AvgCPU,HPAAction" | Out-File -FilePath $ResultsFile -Encoding UTF8

# Results array
$allResults = @()
$testNum = 1

foreach ($t in $tests) {
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "TEST $testNum of $($tests.Count): $($t.Desc)" -ForegroundColor Cyan
    Write-Host "Users: $($t.Users) | Spawn: $($t.Spawn)/s | Duration: $($t.Time)s" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Cyan
    
    # Get pre-test HPA state
    $preHPA = kubectl get hpa inference -o json | ConvertFrom-Json
    $preReplicas = $preHPA.status.currentReplicas
    $preCPU = 0
    if ($preHPA.status.currentMetrics) {
        foreach ($m in $preHPA.status.currentMetrics) {
            if ($m.type -eq "Resource" -and $m.resource.name -eq "cpu") {
                $preCPU = $m.resource.current.averageUtilization
            }
        }
    }
    
    Write-Host "Pre-test: $preReplicas replicas, $preCPU% CPU" -ForegroundColor Yellow
    
    # Run Locust test
    Write-Host "Running Locust test..." -ForegroundColor Cyan
    $cmd = "cd /home/locust && locust -f locustfile.py --headless --host=http://inference:8000 -u $($t.Users) -r $($t.Spawn) -t $($t.Time)s"
    $output = kubectl exec deployment/locust-master -- sh -c $cmd 2>&1 | Out-String
    Write-Host "Test completed, parsing results..." -ForegroundColor DarkGray
    
    # Monitor HPA during test (background sampling)
    $hpaSnaps = @()
    $sampleCount = [Math]::Floor($t.Time / 10)
    for ($i = 0; $i -lt $sampleCount; $i++) {
        Start-Sleep -Seconds 10
        $snap = kubectl get hpa inference -o json | ConvertFrom-Json
        $snapReplicas = $snap.status.currentReplicas
        $snapCPU = 0
        if ($snap.status.currentMetrics) {
            foreach ($m in $snap.status.currentMetrics) {
                if ($m.type -eq "Resource" -and $m.resource.name -eq "cpu") {
                    $snapCPU = $m.resource.current.averageUtilization
                }
            }
        }
        $hpaSnaps += @{Replicas=$snapReplicas; CPU=$snapCPU}
        Write-Host "  HPA: $snapReplicas replicas, $snapCPU% CPU" -ForegroundColor DarkGray
    }
    
    # Get post-test state
    Start-Sleep -Seconds 5
    $postHPA = kubectl get hpa inference -o json | ConvertFrom-Json
    $postReplicas = $postHPA.status.currentReplicas
    $postCPU = 0
    if ($postHPA.status.currentMetrics) {
        foreach ($m in $postHPA.status.currentMetrics) {
            if ($m.type -eq "Resource" -and $m.resource.name -eq "cpu") {
                $postCPU = $m.resource.current.averageUtilization
            }
        }
    }
    
    # Parse Locust output - look for "Aggregated" line
    $totalReqs = 0
    $totalFails = 0
    $avgLatency = 0
    $p95Latency = 0
    $throughput = 0
    
    $lines = $output -split "`n"
    foreach ($line in $lines) {
        # Match "Aggregated" line: "Aggregated  110  0(0.00%) | 137 ..."
        if ($line -match 'Aggregated\s+(\d+)\s+(\d+)\([\d\.]+%\)\s+\|\s+(\d+)') {
            $totalReqs = [int]$Matches[1]
            $totalFails = [int]$Matches[2]
            $avgLatency = [int]$Matches[3]
        }
        # Match throughput from same line: "... | 5.62  0.00"
        if ($line -match 'Aggregated.*\|\s+([\d\.]+)\s+[\d\.]+\s*$') {
            $throughput = [decimal]$Matches[1]
        }
        # Match P95 from percentiles line (95% is 6th column)
        if ($line -match 'Aggregated\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)') {
            $p95Latency = [int]$Matches[6]
        }
    }
    
    # Calculate HPA metrics
    $maxReplicas = $preReplicas
    $cpuSum = 0
    $cpuCount = 0
    foreach ($snap in $hpaSnaps) {
        if ($snap.Replicas -gt $maxReplicas) {
            $maxReplicas = $snap.Replicas
        }
        if ($snap.CPU -gt 0) {
            $cpuSum += $snap.CPU
            $cpuCount++
        }
    }
    $avgCPU = if ($cpuCount -gt 0) { [Math]::Round($cpuSum / $cpuCount, 1) } else { 0 }
    
    $hpaAction = "steady"
    if ($maxReplicas -gt $preReplicas) {
        $hpaAction = "scaled_up"
    } elseif ($postReplicas -lt $preReplicas) {
        $hpaAction = "scaled_down"
    }
    
    # Display results
    Write-Host ""
    Write-Host "Test Complete!" -ForegroundColor Green
    Write-Host "  Requests: $totalReqs | Failures: $totalFails" -ForegroundColor White
    Write-Host "  Throughput: $([Math]::Round($throughput, 2)) req/s" -ForegroundColor White
    Write-Host "  Latency: Avg=$([Math]::Round($avgLatency, 0))ms, P95=$([Math]::Round($p95Latency, 0))ms" -ForegroundColor White
    Write-Host "  Replicas: $preReplicas -> $maxReplicas (max) -> $postReplicas" -ForegroundColor White
    Write-Host "  CPU: $avgCPU% (avg)" -ForegroundColor White
    Write-Host "  HPA Action: $hpaAction" -ForegroundColor $(if ($hpaAction -eq 'scaled_up') { 'Yellow' } else { 'White' })
    Write-Host ""
    
    # Save to CSV
    $csvLine = "$testNum,$($t.Users),$($t.Spawn),$($t.Time),`"$($t.Desc)`",$totalReqs,$totalFails,$([Math]::Round($avgLatency, 1)),$([Math]::Round($p95Latency, 1)),$([Math]::Round($throughput, 2)),$preReplicas,$maxReplicas,$postReplicas,$avgCPU,$hpaAction"
    $csvLine | Out-File -FilePath $ResultsFile -Append -Encoding UTF8
    
    # Store for summary
    $allResults += @{
        TestID=$testNum; Users=$t.Users; Spawn=$t.Spawn; Time=$t.Time; Desc=$t.Desc;
        TotalReqs=$totalReqs; Fails=$totalFails; AvgLat=$avgLatency; P95Lat=$p95Latency;
        Throughput=$throughput; PreReplicas=$preReplicas; MaxReplicas=$maxReplicas;
        PostReplicas=$postReplicas; AvgCPU=$avgCPU; HPAAction=$hpaAction
    }
    
    $testNum++
    
    # Cool-down
    if ($testNum -le $tests.Count) {
        Write-Host "Cool-down: 30 seconds..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 30
    }
}

# Summary
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "PERFORMANCE SUITE SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$fmt = "{0,-6} {1,-7} {2,-9} {3,-12} {4,-12} {5,-8} {6,-8} {7,-10}"
Write-Host ($fmt -f "USERS", "SPAWN", "TIME(s)", "REPLICAS", "THROUGHPUT", "P95(ms)", "CPU(%)", "HPA_ACTION") -ForegroundColor Yellow

foreach ($r in $allResults) {
    $repDisplay = "$($r.PreReplicas)->$($r.MaxReplicas)"
    $tputDisplay = [Math]::Round($r.Throughput, 1)
    $p95Display = [Math]::Round($r.P95Lat, 0)
    
    Write-Host ($fmt -f $r.Users, $r.Spawn, $r.Time, $repDisplay, $tputDisplay, $p95Display, $r.AvgCPU, $r.HPAAction)
}

# Validation
Write-Host ""
Write-Host "VALIDATION RESULTS:" -ForegroundColor Yellow

$totalFails = ($allResults | Measure-Object -Property Fails -Sum).Sum
$maxP95 = ($allResults | Measure-Object -Property P95Lat -Maximum).Maximum
$scaledTests = ($allResults | Where-Object { $_.HPAAction -eq 'scaled_up' }).Count

Write-Host "  Total Failures: $totalFails (Target: 0)" -ForegroundColor $(if ($totalFails -eq 0) { 'Green' } else { 'Red' })
Write-Host "  Max P95 Latency: $([Math]::Round($maxP95, 0))ms (Target: < 1000ms)" -ForegroundColor $(if ($maxP95 -lt 1000) { 'Green' } else { 'Red' })
Write-Host "  HPA Scale-up Events: $scaledTests / $($allResults.Count) tests" -ForegroundColor Cyan

Write-Host ""
Write-Host "Results saved to: $ResultsFile" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
