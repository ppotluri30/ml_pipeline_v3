# Simple HPA Performance Test Runner
# Runs Locust tests and collects basic metrics

param([string]$OutputDir = ".\reports\hpa_performance_simple")

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$ResultsFile = Join-Path $OutputDir "results.csv"
$LogFile = Join-Path $OutputDir "test.log"

# Test scenarios
$tests = @(
    @{Users=10; Spawn=2; Time=30; Desc="Light baseline"},
    @{Users=25; Spawn=5; Time=60; Desc="Medium load"},
    @{Users=50; Spawn=10; Time=60; Desc="Heavy load"}
)

# Write CSV header
"TestID,Users,SpawnRate,Duration,Description,TotalRequests,Failures,AvgLatency,P95Latency,Throughput" | Out-File -FilePath $ResultsFile -Encoding UTF8

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "HPA PERFORMANCE TEST - Simple Version" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
Write-Host "Running $($tests.Count) test scenarios`n" -ForegroundColor Yellow

$testNum = 0
foreach ($t in $tests) {
    $testNum++
    Write-Host "TEST $testNum of $($tests.Count): $($t.Desc)" -ForegroundColor Green
    Write-Host "Users: $($t.Users) | Spawn: $($t.Spawn)/s | Duration: $($t.Time)s" -ForegroundColor Gray
    
    # Run test - use call operator for proper execution
    $cmd = "cd /home/locust && locust -f locustfile.py --headless --host=http://inference:8000 -u $($t.Users) -r $($t.Spawn) -t $($t.Time)s"
    Write-Host "Executing Locust..." -ForegroundColor DarkGray
    
    try {
        $result = & kubectl exec deployment/locust-master -- sh -c $cmd 2>&1
        $output = $result | Out-String
        Write-Host "Test completed, parsing results..." -ForegroundColor DarkGray
    } catch {
        Write-Host "ERROR running test: $($_.Exception.Message)" -ForegroundColor Red
        continue
    }
    
    # Parse output - find the final Aggregated line
    $totalReqs = 0
    $totalFails = 0
    $avgLatency = 0
    $p95Latency = 0
    $throughput = 0
    
    # Match: "Aggregated   110   0(0.00%) |  137   46  382  120 |  5.62   0.00"
    if ($output -match 'Aggregated\s+(\d+)\s+(\d+)\([^\)]+\)\s+\|\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\|\s+([\d\.]+)') {
        $totalReqs = [int]$Matches[1]
        $totalFails = [int]$Matches[2]
        $avgLatency = [int]$Matches[3]
        $throughput = [decimal]$Matches[4]
        Write-Host "  Requests: $totalReqs | Failures: $totalFails | Avg Latency: ${avgLatency}ms | Throughput: $throughput req/s" -ForegroundColor White
    } else {
        Write-Host "  WARNING: Could not parse stats from output" -ForegroundColor Yellow
    }
    
    # Match P95 from percentiles table:  "Aggregated   97  110  110  120  140  150  200  200  200  200  200   30"
    # The 95% column is the 6th value
    if ($output -match 'Aggregated\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)') {
        $p95Latency = [int]$Matches[1]
        Write-Host "  P95 Latency: ${p95Latency}ms" -ForegroundColor White
    }
    
    # Write to CSV
    $csvLine = "$testNum,$($t.Users),$($t.Spawn),$($t.Time),`"$($t.Desc)`",$totalReqs,$totalFails,$avgLatency,$p95Latency,$throughput"
    $csvLine | Out-File -FilePath $ResultsFile -Append -Encoding UTF8
    
    Write-Host "" # blank line
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ALL TESTS COMPLETE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Results saved to: $ResultsFile`n" -ForegroundColor Yellow

# Display results
Write-Host "SUMMARY:" -ForegroundColor Cyan
Import-Csv $ResultsFile | Format-Table -AutoSize
