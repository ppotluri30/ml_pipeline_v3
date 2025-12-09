<#
.SYNOPSIS
    Automated Distributed Locust Test Runner with Live Updates

.DESCRIPTION
    Runs comprehensive load tests across all combinations of:
    - Inference replicas: 1, 2, 4, 8
    - Locust workers: 4, 8
    - User counts: 200, 400, 800
    
    Keeps all containers persistent and provides live monitoring.

.PARAMETER TestDuration
    Duration of each test in seconds (default: 120)

.PARAMETER ReplicaCounts
    Array of inference replica counts to test (default: @(1,2,4,8))

.PARAMETER WorkerCounts
    Array of Locust worker counts to test (default: @(4,8))

.PARAMETER UserCounts
    Array of user counts to test (default: @(200,400,800))

.EXAMPLE
    .\run_all_locust_tests.ps1
    
.EXAMPLE
    .\run_all_locust_tests.ps1 -TestDuration 180 -UserCounts @(100,200,400)
#>

[CmdletBinding()]
param(
    [int]$TestDuration = 120,
    [int[]]$ReplicaCounts = @(1, 2, 4, 8),
    [int[]]$WorkerCounts = @(4, 8),
    [int[]]$UserCounts = @(200, 400, 800),
    [string]$ResultsBaseDir = "locust/results/auto_matrix",
    [string]$LocustMasterUrl = "http://localhost:8089"
)

# Color output functions
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message) Write-ColorOutput "[OK] $Message" "Green" }
function Write-Info { param([string]$Message) Write-ColorOutput "[INFO] $Message" "Cyan" }
function Write-Warning { param([string]$Message) Write-ColorOutput "[WARN] $Message" "Yellow" }
function Write-Error { param([string]$Message) Write-ColorOutput "[ERROR] $Message" "Red" }
function Write-Running { param([string]$Message) Write-ColorOutput "[RUN] $Message" "Yellow" }

# Results tracking
$script:AllResults = @()
$script:SummaryFile = Join-Path $ResultsBaseDir "auto_summary.csv"
$script:SummaryMdFile = Join-Path $ResultsBaseDir "auto_summary.md"

function Initialize-ResultsDirectory {
    Write-Info "Creating results directory: $ResultsBaseDir"
    New-Item -ItemType Directory -Force -Path $ResultsBaseDir | Out-Null
    
    # Initialize CSV with headers
    "Replicas,Workers,Users,RPS,Median_ms,P95_ms,P99_ms,Failures_Pct,Total_Requests,Duration_s" | 
        Out-File -FilePath $script:SummaryFile -Encoding UTF8
}

function Wait-ForLocustReady {
    Write-Info "Waiting for Locust master to be ready..."
    $maxAttempts = 30
    for ($i = 1; $i -le $maxAttempts; $i++) {
        try {
            $response = Invoke-RestMethod -Uri "$LocustMasterUrl/" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response) {
                Write-Success "Locust master is ready"
                return $true
            }
        } catch {
            # Silently retry
        }
        Start-Sleep -Seconds 2
    }
    Write-Error "Locust master not ready after $($maxAttempts * 2) seconds"
    return $false
}

function Scale-InferenceContainers {
    param([int]$ReplicaCount)
    
    Write-Info "Scaling inference to $ReplicaCount replicas..."
    try {
        $result = docker compose up -d --scale inference=$ReplicaCount --no-recreate inference inference-lb 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Scaled to $ReplicaCount inference replicas"
            Start-Sleep -Seconds 12  # Wait for health checks
            return $true
        } else {
            Write-Error "Failed to scale inference: $result"
            return $false
        }
    } catch {
        Write-Error "Exception scaling inference: $_"
        return $false
    }
}

function Scale-LocustWorkers {
    param([int]$WorkerCount)
    
    Write-Info "Scaling Locust workers to $WorkerCount..."
    try {
        $result = docker compose up -d --scale locust-worker=$WorkerCount locust-worker 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Scaled to $WorkerCount Locust workers"
            Start-Sleep -Seconds 8  # Wait for workers to connect
            return $true
        } else {
            Write-Error "Failed to scale workers: $result"
            return $false
        }
    } catch {
        Write-Error "Exception scaling workers: $_"
        return $false
    }
}

function Get-LocustStats {
    try {
        $stats = Invoke-RestMethod -Uri "$LocustMasterUrl/stats/requests" -TimeoutSec 5 -ErrorAction Stop
        return $stats
    } catch {
        return $null
    }
}

function Start-LocustTest {
    param(
        [int]$Users,
        [int]$SpawnRate
    )
    
    try {
        $body = @{
            user_count = $Users
            spawn_rate = $SpawnRate
            host = "http://inference-lb"
        }
        
        $response = Invoke-RestMethod -Uri "$LocustMasterUrl/swarm" -Method Post -Body $body -TimeoutSec 10
        return $true
    } catch {
        Write-Error "Failed to start test: $_"
        return $false
    }
}

function Stop-LocustTest {
    try {
        Invoke-RestMethod -Uri "$LocustMasterUrl/stop" -TimeoutSec 10 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Reset-LocustStats {
    try {
        Invoke-RestMethod -Uri "$LocustMasterUrl/stats/reset" -TimeoutSec 10 | Out-Null
        Start-Sleep -Seconds 2
        return $true
    } catch {
        return $false
    }
}

function Show-LiveStats {
    param(
        [int]$Replicas,
        [int]$Workers,
        [int]$Users,
        [int]$ElapsedSeconds,
        [int]$TotalSeconds
    )
    
    $stats = Get-LocustStats
    if ($stats) {
        $aggregated = $stats.stats | Where-Object { $_.name -eq "Aggregated" } | Select-Object -First 1
        if ($aggregated) {
            $rps = [math]::Round($aggregated.current_rps, 2)
            $median = $aggregated.median_response_time
            $p95 = if ($aggregated.PSObject.Properties['response_time_percentile_0.95']) { $aggregated.'response_time_percentile_0.95' } else { $aggregated.avg_response_time }
            $requests = $aggregated.num_requests
            $failures = $aggregated.num_failures
            
            $progress = [math]::Round(($ElapsedSeconds / $TotalSeconds) * 100)
            $progressBar = "█" * [math]::Floor($progress / 5) + "░" * (20 - [math]::Floor($progress / 5))
            
            Write-Host "`r[Replicas=$Replicas Workers=$Workers Users=$Users] " -NoNewline -ForegroundColor Cyan
            Write-Host "[$progressBar] $progress% " -NoNewline -ForegroundColor White
            Write-Host "| RPS: $rps | Med: ${median}ms | P95: ${p95}ms | Reqs: $requests | Fails: $failures" -NoNewline -ForegroundColor Yellow
        }
    }
}

function Run-SingleTest {
    param(
        [int]$Replicas,
        [int]$Workers,
        [int]$Users,
        [int]$Duration
    )
    
    $testName = "replicas${Replicas}_workers${Workers}_u${Users}"
    $testDir = Join-Path $ResultsBaseDir $testName
    
    Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
    Write-ColorOutput "[TEST] $testName" "Magenta"
    Write-ColorOutput "   Inference Replicas: $Replicas" "White"
    Write-ColorOutput "   Locust Workers: $Workers" "White"
    Write-ColorOutput "   Concurrent Users: $Users" "White"
    Write-ColorOutput "   Duration: ${Duration}s" "White"
    Write-Host ("=" * 80) -ForegroundColor Cyan
    
    # Create test results directory
    New-Item -ItemType Directory -Force -Path $testDir | Out-Null
    
    # Reset stats before test
    Reset-LocustStats | Out-Null
    
    # Calculate spawn rate (aim for ramping up in 10-20 seconds)
    $spawnRate = [math]::Max([math]::Floor($Users / 15), 10)
    $rampUpTime = [math]::Ceiling($Users / $spawnRate)
    
    Write-Running "Starting test: $Users users at ${spawnRate}/s spawn rate..."
    if (-not (Start-LocustTest -Users $Users -SpawnRate $spawnRate)) {
        Write-Error "Failed to start test for $testName"
        return $null
    }
    
    Write-Running "Ramping up users (${rampUpTime}s)..."
    Start-Sleep -Seconds ($rampUpTime + 3)
    
    Write-Running "Test running for ${Duration}s with live monitoring..."
    $startTime = Get-Date
    $endTime = $startTime.AddSeconds($Duration)
    $updateInterval = 15
    
    while ((Get-Date) -lt $endTime) {
        $elapsed = [int]((Get-Date) - $startTime).TotalSeconds
        Show-LiveStats -Replicas $Replicas -Workers $Workers -Users $Users -ElapsedSeconds $elapsed -TotalSeconds $Duration
        Start-Sleep -Seconds 1
        
        # Update every interval or at end
        if ($elapsed % $updateInterval -eq 0 -or ((Get-Date) -ge $endTime.AddSeconds(-1))) {
            Write-Host ""  # New line after progress
        }
    }
    
    Write-Host ""  # Final newline
    Write-Running "Test duration complete, collecting final stats..."
    Start-Sleep -Seconds 3
    
    # Get final statistics
    $finalStats = Get-LocustStats
    if (-not $finalStats) {
        Write-Error "Failed to retrieve final stats"
        Stop-LocustTest | Out-Null
        return $null
    }
    
    $aggregated = $finalStats.stats | Where-Object { $_.name -eq "Aggregated" } | Select-Object -First 1
    if (-not $aggregated) {
        Write-Error "No aggregated stats found"
        Stop-LocustTest | Out-Null
        return $null
    }
    
    # Extract metrics with fallback for percentiles
    $p95Value = if ($aggregated.PSObject.Properties['response_time_percentile_0.95']) { $aggregated.'response_time_percentile_0.95' } else { $aggregated.avg_response_time }
    $p99Value = if ($aggregated.PSObject.Properties['response_time_percentile_0.99']) { $aggregated.'response_time_percentile_0.99' } else { $aggregated.max_response_time }
    
    $result = [PSCustomObject]@{
        Replicas = $Replicas
        Workers = $Workers
        Users = $Users
        RPS = [math]::Round($aggregated.current_rps, 2)
        Median_ms = $aggregated.median_response_time
        P95_ms = $p95Value
        P99_ms = $p99Value
        Failures_Pct = [math]::Round(($aggregated.num_failures / [math]::Max($aggregated.num_requests, 1)) * 100, 2)
        Total_Requests = $aggregated.num_requests
        Duration_s = $Duration
    }
    
    # Display results
    Write-Host "`n[RESULTS] Test Results:" -ForegroundColor Green
    Write-Host "   Total Requests:  $($result.Total_Requests)" -ForegroundColor White
    Write-Host "   RPS:             $($result.RPS)" -ForegroundColor White
    Write-Host "   Median:          $($result.Median_ms)ms" -ForegroundColor White
    Write-Host "   P95:             $($result.P95_ms)ms" -ForegroundColor White
    Write-Host "   P99:             $($result.P99_ms)ms" -ForegroundColor White
    Write-Host "   Failures:        $($aggregated.num_failures) ($($result.Failures_Pct)%)" -ForegroundColor $(if ($result.Failures_Pct -gt 0) { "Red" } else { "Green" })
    
    # Stop test
    Stop-LocustTest | Out-Null
    Start-Sleep -Seconds 5
    
    Write-Success "Test completed: $testName"
    return $result
}

function Export-SummaryMarkdown {
    param([array]$Results)
    
    $markdown = @"
# Distributed Locust Load Testing Results

**Test Date**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")  
**Test Duration**: ${TestDuration}s per configuration  
**Total Tests**: $($Results.Count)

## Configuration Matrix

- **Inference Replicas**: $($ReplicaCounts -join ', ')
- **Locust Workers**: $($WorkerCounts -join ', ')
- **User Counts**: $($UserCounts -join ', ')

## Complete Results

| Replicas | Workers | Users | RPS | Median (ms) | P95 (ms) | P99 (ms) | Failures (%) | Total Requests |
|----------|---------|-------|-----|-------------|----------|----------|--------------|----------------|
"@

    foreach ($result in $Results | Sort-Object Replicas, Workers, Users) {
        $markdown += "`n| $($result.Replicas) | $($result.Workers) | $($result.Users) | $($result.RPS) | $($result.Median_ms) | $($result.P95_ms) | $($result.P99_ms) | $($result.Failures_Pct) | $($result.Total_Requests) |"
    }
    
    # Add analysis sections
    $markdown += @"

## Performance Analysis

### Throughput Scaling by Replica Count

"@
    
    foreach ($workers in $WorkerCounts) {
        $markdown += "`n#### $workers Locust Workers`n`n"
        $markdown += "| Replicas | 200 Users RPS | 400 Users RPS | 800 Users RPS |`n"
        $markdown += "|----------|---------------|---------------|---------------|`n"
        
        foreach ($replicas in $ReplicaCounts) {
            $rps200 = ($Results | Where-Object { $_.Replicas -eq $replicas -and $_.Workers -eq $workers -and $_.Users -eq 200 }).RPS
            $rps400 = ($Results | Where-Object { $_.Replicas -eq $replicas -and $_.Workers -eq $workers -and $_.Users -eq 400 }).RPS
            $rps800 = ($Results | Where-Object { $_.Replicas -eq $replicas -and $_.Workers -eq $workers -and $_.Users -eq 800 }).RPS
            $markdown += "| $replicas | $rps200 | $rps400 | $rps800 |`n"
        }
    }
    
    $markdown += @"

### Latency Analysis

Best P95 latencies achieved:

"@
    
    $bestP95 = $Results | Sort-Object P95_ms | Select-Object -First 5
    foreach ($result in $bestP95) {
        $markdown += "- **$($result.P95_ms)ms** @ $($result.Replicas) replicas, $($result.Workers) workers, $($result.Users) users`n"
    }
    
    $markdown += @"

### Key Findings

1. **Optimal Replica Count**: $(($Results | Sort-Object RPS -Descending | Select-Object -First 1).Replicas) replicas achieved highest RPS
2. **Maximum Throughput**: $(($Results | Sort-Object RPS -Descending | Select-Object -First 1).RPS) req/s
3. **Best Latency**: $(($Results | Sort-Object P95_ms | Select-Object -First 1).P95_ms)ms P95
4. **Failure Rate**: $(if (($Results | Where-Object { $_.Failures_Pct -gt 0 }).Count -eq 0) { "0% across all tests ✅" } else { "Issues detected in some configurations ⚠️" })

---

*Generated by run_all_locust_tests.ps1*
"@
    
    $markdown | Out-File -FilePath $script:SummaryMdFile -Encoding UTF8
    Write-Success "Markdown summary saved to: $script:SummaryMdFile"
}

function Show-FinalSummary {
    param([array]$Results)
    
    Write-Host "`n`n" + ("=" * 100) -ForegroundColor Green
    Write-ColorOutput "[COMPLETE] ALL TESTS COMPLETED!" "Green"
    Write-Host ("=" * 100) -ForegroundColor Green
    
    Write-Host "`n[SUMMARY] FINAL SUMMARY TABLE`n" -ForegroundColor Cyan
    
    # Create formatted table
    $tableHeader = "| {0,-10} | {1,-8} | {2,-6} | {3,-8} | {4,-12} | {5,-10} | {6,-10} | {7,-12} |" -f `
        "Replicas", "Workers", "Users", "RPS", "Median (ms)", "P95 (ms)", "P99 (ms)", "Fail (%)"
    $tableSeparator = "+" + ("-" * 11) + "+" + ("-" * 9) + "+" + ("-" * 7) + "+" + ("-" * 9) + "+" + ("-" * 13) + "+" + ("-" * 11) + "+" + ("-" * 11) + "+" + ("-" * 13) + "+"
    
    Write-Host $tableSeparator -ForegroundColor White
    Write-Host $tableHeader -ForegroundColor Yellow
    Write-Host $tableSeparator -ForegroundColor White
    
    foreach ($result in $Results | Sort-Object Replicas, Workers, Users) {
        $color = if ($result.Failures_Pct -gt 0) { "Red" } elseif ($result.RPS -gt 50) { "Green" } else { "White" }
        $tableRow = "| {0,-10} | {1,-8} | {2,-6} | {3,-8} | {4,-12} | {5,-10} | {6,-10} | {7,-12} |" -f `
            $result.Replicas, $result.Workers, $result.Users, $result.RPS, `
            $result.Median_ms, $result.P95_ms, $result.P99_ms, $result.Failures_Pct
        Write-Host $tableRow -ForegroundColor $color
    }
    
    Write-Host $tableSeparator -ForegroundColor White
    
    # Statistics
    Write-Host "`n[STATS] Key Statistics:" -ForegroundColor Cyan
    Write-Host "   Total Tests Run:      $($Results.Count)" -ForegroundColor White
    Write-Host "   Highest RPS:          $(($Results | Sort-Object RPS -Descending | Select-Object -First 1).RPS)" -ForegroundColor Green
    Write-Host "   Best P95 Latency:     $(($Results | Sort-Object P95_ms | Select-Object -First 1).P95_ms)ms" -ForegroundColor Green
    Write-Host "   Total Requests:       $(($Results | Measure-Object -Property Total_Requests -Sum).Sum)" -ForegroundColor White
    
    $failedTests = $Results | Where-Object { $_.Failures_Pct -gt 0 }
    if ($failedTests.Count -eq 0) {
        Write-Host "   Failed Tests:         0 [PASS]" -ForegroundColor Green
    } else {
        Write-Host "   Failed Tests:         $($failedTests.Count) [WARN]" -ForegroundColor Red
    }
    
    Write-Host "`n[FILES] Results Location:" -ForegroundColor Cyan
    Write-Host "   Directory:            $ResultsBaseDir" -ForegroundColor White
    Write-Host "   CSV Summary:          $script:SummaryFile" -ForegroundColor White
    Write-Host "   Markdown Report:      $script:SummaryMdFile" -ForegroundColor White
    
    Write-Host "`n" + ("=" * 100) -ForegroundColor Green
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function Main {
    Clear-Host
    
    Write-Host @"
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              DISTRIBUTED LOCUST AUTOMATED TEST SUITE                       ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

    Write-Host "`n[CONFIG] Test Configuration:" -ForegroundColor Yellow
    Write-Host "   Duration per test:    ${TestDuration}s" -ForegroundColor White
    Write-Host "   Inference replicas:   $($ReplicaCounts -join ', ')" -ForegroundColor White
    Write-Host "   Locust workers:       $($WorkerCounts -join ', ')" -ForegroundColor White
    Write-Host "   User counts:          $($UserCounts -join ', ')" -ForegroundColor White
    Write-Host "   Total tests planned:  $(($ReplicaCounts.Count * $WorkerCounts.Count * $UserCounts.Count))" -ForegroundColor Green
    
    # Initialize
    Initialize-ResultsDirectory
    
    # Wait for Locust master
    if (-not (Wait-ForLocustReady)) {
        Write-Error "Cannot proceed without Locust master"
        return
    }
    
    # Main test loop
    $testCounter = 0
    $totalTests = $ReplicaCounts.Count * $WorkerCounts.Count * $UserCounts.Count
    
    foreach ($workers in $WorkerCounts) {
        Write-Host "`n" + ("=" * 100) -ForegroundColor Magenta
        Write-ColorOutput "[CONFIG] SWITCHING TO $workers LOCUST WORKERS" "Magenta"
        Write-Host ("=" * 100) -ForegroundColor Magenta
        
        if (-not (Scale-LocustWorkers -WorkerCount $workers)) {
            Write-Error "Failed to scale to $workers workers, skipping this worker tier"
            continue
        }
        
        Start-Sleep -Seconds 10  # Wait for workers to stabilize
        
        foreach ($replicas in $ReplicaCounts) {
            if (-not (Scale-InferenceContainers -ReplicaCount $replicas)) {
                Write-Error "Failed to scale to $replicas replicas, skipping"
                continue
            }
            
            foreach ($users in $UserCounts) {
                $testCounter++
                Write-Info "Progress: Test $testCounter of $totalTests"
                
                $result = Run-SingleTest -Replicas $replicas -Workers $workers -Users $users -Duration $TestDuration
                
                if ($result) {
                    $script:AllResults += $result
                    
                    # Append to CSV
                    "$($result.Replicas),$($result.Workers),$($result.Users),$($result.RPS),$($result.Median_ms),$($result.P95_ms),$($result.P99_ms),$($result.Failures_Pct),$($result.Total_Requests),$($result.Duration_s)" | 
                        Out-File -FilePath $script:SummaryFile -Append -Encoding UTF8
                }
                
                # Cool down between tests
                if ($testCounter -lt $totalTests) {
                    Write-Info "Cooling down for 15 seconds..."
                    Start-Sleep -Seconds 15
                }
            }
        }
    }
    
    # Generate final outputs
    if ($script:AllResults.Count -gt 0) {
        Export-SummaryMarkdown -Results $script:AllResults
        Show-FinalSummary -Results $script:AllResults
    } else {
        Write-Error "No test results collected!"
    }
}

# Run the main function
try {
    Main
} catch {
    Write-Error "Fatal error: $_"
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
} finally {
    Write-Host "`nScript execution finished." -ForegroundColor Cyan

}
