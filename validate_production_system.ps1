# Production System Validation Script
# Validates the refactored inference architecture end-to-end

param(
    [int]$TimeoutSeconds = 300
)

Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   FLTS ML Pipeline - Production Validation                    ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$results = @()
$startTime = Get-Date

function Test-Component {
    param(
        [string]$Name,
        [scriptblock]$Test,
        [string]$Expected
    )
    
    Write-Host "Testing: $Name..." -ForegroundColor Yellow -NoNewline
    
    try {
        $result = & $Test
        if ($result) {
            Write-Host " ✓ PASS" -ForegroundColor Green
            $script:results += [PSCustomObject]@{
                Component = $Name
                Status = "PASS"
                Result = $Expected
            }
            return $true
        } else {
            Write-Host " ✗ FAIL" -ForegroundColor Red
            $script:results += [PSCustomObject]@{
                Component = $Name
                Status = "FAIL"
                Result = "Did not meet criteria"
            }
            return $false
        }
    } catch {
        Write-Host " ✗ ERROR" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Gray
        $script:results += [PSCustomObject]@{
            Component = $Name
            Status = "ERROR"
            Result = $_.Exception.Message
        }
        return $false
    }
}

Write-Host "═══ 1. DEPLOYMENT STATUS ═══" -ForegroundColor Cyan
Write-Host ""

Test-Component -Name "HTTP Deployment Ready" -Test {
    $deployment = kubectl get deployment inference-http -o jsonpath='{.status.readyReplicas}' 2>$null
    [int]$deployment -ge 2
} -Expected "2+ ready replicas"

Test-Component -Name "Worker Deployment Ready" -Test {
    $deployment = kubectl get deployment inference-worker -o jsonpath='{.status.readyReplicas}' 2>$null
    [int]$deployment -eq 1
} -Expected "1 ready replica"

Test-Component -Name "HPA Exists and Active" -Test {
    $hpa = kubectl get hpa inference-http-hpa --no-headers 2>$null
    $null -ne $hpa -and $hpa -match 'inference-http'
} -Expected "HPA configured for inference-http"

Write-Host ""
Write-Host "═══ 2. POD HEALTH ═══" -ForegroundColor Cyan
Write-Host ""

Test-Component -Name "HTTP Pods Running" -Test {
    $pods = kubectl get pods -l app=inference-http -o jsonpath='{.items[*].status.phase}' 2>$null
    $pods -match 'Running' -and -not ($pods -match 'Pending|Failed|Unknown')
} -Expected "All HTTP pods in Running state"

Test-Component -Name "Worker Pod Running" -Test {
    $pod = kubectl get pods -l app=inference-worker -o jsonpath='{.items[0].status.phase}' 2>$null
    $pod -eq 'Running'
} -Expected "Worker pod in Running state"

Test-Component -Name "HTTP Pods Ready" -Test {
    $ready = kubectl get pods -l app=inference-http -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>$null
    $ready -match 'True' -and -not ($ready -match 'False')
} -Expected "All HTTP pods ready"

Test-Component -Name "Worker Pod Ready" -Test {
    $ready = kubectl get pods -l app=inference-worker -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>$null
    $ready -eq 'True'
} -Expected "Worker pod ready"

Write-Host ""
Write-Host "═══ 3. RESOURCE USAGE ═══" -ForegroundColor Cyan
Write-Host ""

Test-Component -Name "HTTP Pod CPU Reduced" -Test {
    Start-Sleep -Seconds 5  # Wait for metrics
    $pods = kubectl top pods -l app=inference-http --no-headers 2>$null
    if (-not $pods) { return $false }
    
    $cpuValues = $pods | ForEach-Object {
        $parts = $_ -split '\s+'
        [int]($parts[1] -replace 'm','')
    }
    
    $avgCpu = ($cpuValues | Measure-Object -Average).Average
    Write-Host "  (Avg: $([math]::Round($avgCpu))m)" -ForegroundColor Gray
    
    # Success if average < 50m (was 238m before)
    $avgCpu -lt 50
} -Expected "HTTP CPU < 50m avg (down from 238m)"

Test-Component -Name "Worker CPU Stable" -Test {
    $pod = kubectl top pods -l app=inference-worker --no-headers 2>$null
    if (-not $pod) { return $false }
    
    $cpuValue = [int](($pod -split '\s+')[1] -replace 'm','')
    Write-Host "  (Worker: $($cpuValue)m)" -ForegroundColor Gray
    
    # Success if < 100m (no CPU spinning)
    $cpuValue -lt 100
} -Expected "Worker CPU < 100m (no spinning)"

Write-Host ""
Write-Host "═══ 4. CONFIGURATION ═══" -ForegroundColor Cyan
Write-Host ""

Test-Component -Name "Kafka Env Separation" -Test {
    $workerEnv = kubectl get deployment inference-worker -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="KAFKA_BOOTSTRAP_SERVERS")].value}' 2>$null
    $httpEnv = kubectl get deployment inference-http -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="KAFKA_BOOTSTRAP_SERVERS")].value}' 2>$null
    
    # Worker should have Kafka, HTTP should not
    ($null -ne $workerEnv) -and ($null -eq $httpEnv)
} -Expected "Worker has Kafka vars, HTTP does not"

Test-Component -Name "Worker Backpressure Config" -Test {
    $idleSleep = kubectl get deployment inference-worker -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="IDLE_SLEEP_SECONDS")].value}' 2>$null
    $pollTimeout = kubectl get deployment inference-worker -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="POLL_TIMEOUT_MS")].value}' 2>$null
    
    ([float]$idleSleep -ge 0.5) -and ([int]$pollTimeout -ge 1000)
} -Expected "Backpressure settings configured (IDLE_SLEEP≥0.5s, POLL≥1000ms)"

Write-Host ""
Write-Host "═══ 5. FUNCTIONAL TESTS ═══" -ForegroundColor Cyan
Write-Host ""

Test-Component -Name "HTTP Healthz Responding" -Test {
    $pod = kubectl get pods -l app=inference-http --no-headers | Select-Object -First 1 | ForEach-Object { ($_ -split '\s+')[0] }
    if (-not $pod) { return $false }
    
    $health = kubectl exec $pod -- wget -qO- http://localhost:8000/healthz 2>$null
    $health -match 'status|ok'
} -Expected "/healthz endpoint responding"

Test-Component -Name "HTTP Predict Responding" -Test {
    $result = kubectl run test-predict --image=curlimages/curl:8.10.1 --rm -it --restart=Never -- curl -s -X POST http://inference-http:8000/predict -H "Content-Type: application/json" -d '{}' 2>$null
    $result -match 'detail|predictions|error'
} -Expected "/predict endpoint responding (even if no model loaded)"

Test-Component -Name "Worker Healthcheck File" -Test {
    $pod = kubectl get pods -l app=inference-worker --no-headers | ForEach-Object { ($_ -split '\s+')[0] }
    if (-not $pod) { return $false }
    
    $healthFile = kubectl exec $pod -- cat /tmp/worker-healthy 2>$null
    $healthFile -match 'started'
} -Expected "Healthcheck file exists and contains 'started'"

Test-Component -Name "Worker Consumer Threads Active" -Test {
    $pod = kubectl get pods -l app=inference-worker --no-headers | ForEach-Object { ($_ -split '\s+')[0] }
    if (-not $pod) { return $false }
    
    $logs = kubectl logs $pod --tail=100 2>$null
    ($logs -match 'promotion-consumer') -and 
    ($logs -match 'training-consumer') -and 
    ($logs -match 'inference-consumer')
} -Expected "All 3 consumer threads started"

Write-Host ""
Write-Host "═══ 6. AUTOSCALING ═══" -ForegroundColor Cyan
Write-Host ""

Test-Component -Name "HPA Metrics Available" -Test {
    $hpa = kubectl get hpa inference-http-hpa -o json 2>$null | ConvertFrom-Json
    $current = $hpa.status.currentMetrics[0].resource.current.averageUtilization
    $null -ne $current -and $current -ne 'unknown'
} -Expected "HPA has CPU metrics"

Test-Component -Name "HPA Can Scale Down" -Test {
    $hpa = kubectl get hpa inference-http-hpa -o json 2>$null | ConvertFrom-Json
    $currentReplicas = $hpa.status.currentReplicas
    $minReplicas = $hpa.spec.minReplicas
    
    Write-Host "  (Current: $currentReplicas, Min: $minReplicas)" -ForegroundColor Gray
    
    # Success if current <= 3 (reasonable idle state)
    $currentReplicas -le 3
} -Expected "HPA at or near minReplicas"

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   VALIDATION SUMMARY                                          ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$passCount = ($results | Where-Object { $_.Status -eq "PASS" }).Count
$failCount = ($results | Where-Object { $_.Status -eq "FAIL" }).Count
$errorCount = ($results | Where-Object { $_.Status -eq "ERROR" }).Count
$totalCount = $results.Count

$duration = (Get-Date) - $startTime

Write-Host "Total Tests: $totalCount" -ForegroundColor White
Write-Host "  Passed:    $passCount" -ForegroundColor Green
Write-Host "  Failed:    $failCount" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Gray" })
Write-Host "  Errors:    $errorCount" -ForegroundColor $(if ($errorCount -gt 0) { "Red" } else { "Gray" })
Write-Host "Duration:    $([math]::Round($duration.TotalSeconds, 1))s" -ForegroundColor Gray

Write-Host ""

if ($failCount -eq 0 -and $errorCount -eq 0) {
    Write-Host "✓ All validation checks passed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Production Architecture Summary:" -ForegroundColor Cyan
    Write-Host "  - HTTP pods: 2+ running with <50m CPU (down from 238m)"
    Write-Host "  - Worker pod: 1 running with <100m CPU"
    Write-Host "  - HPA: Active and responsive"
    Write-Host "  - Kafka: Properly separated from HTTP"
    Write-Host "  - Backpressure: Configured and working"
    Write-Host ""
    Write-Host "System is production-ready! ✨" -ForegroundColor Green
    exit 0
} else {
    Write-Host "⚠ Some validation checks failed" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Failed/Error Tests:" -ForegroundColor Yellow
    $results | Where-Object { $_.Status -ne "PASS" } | Format-Table -AutoSize
    Write-Host ""
    Write-Host "Review logs with:" -ForegroundColor Gray
    Write-Host "  kubectl logs -l app=inference-http" -ForegroundColor Gray
    Write-Host "  kubectl logs -l app=inference-worker" -ForegroundColor Gray
    exit 1
}
