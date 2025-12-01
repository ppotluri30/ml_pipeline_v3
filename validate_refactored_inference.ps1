# Validation script for refactored inference architecture
# Verifies HTTP-only and Kafka worker separation

param(
    [int]$WaitSeconds = 60
)

Write-Host "Validating refactored inference architecture..." -ForegroundColor Cyan
Write-Host "Waiting $WaitSeconds seconds for stabilization...`n"
Start-Sleep -Seconds $WaitSeconds

$validationResults = @()

function Test-ValidationStep {
    param(
        [string]$Name,
        [scriptblock]$Test,
        [string]$SuccessMessage,
        [string]$FailureMessage
    )
    
    Write-Host "Testing: $Name..." -ForegroundColor Yellow
    $result = & $Test
    
    if ($result) {
        Write-Host "  ✓ $SuccessMessage" -ForegroundColor Green
        $script:validationResults += [PSCustomObject]@{
            Test = $Name
            Status = "PASS"
            Message = $SuccessMessage
        }
        return $true
    } else {
        Write-Host "  ✗ $FailureMessage" -ForegroundColor Red
        $script:validationResults += [PSCustomObject]@{
            Test = $Name
            Status = "FAIL"
            Message = $FailureMessage
        }
        return $false
    }
}

Write-Host "=== Deployment Status Checks ===" -ForegroundColor Cyan

# Test 1: HTTP deployment exists and has pods
Test-ValidationStep -Name "HTTP Deployment Running" -Test {
    $deployment = kubectl get deployment inference-http --no-headers 2>$null
    $ready = ($deployment -split '\s+')[1]
    $ready -match '^\d+/\d+$' -and ($ready -split '/')[0] -gt 0
} -SuccessMessage "inference-http deployment is running with ready pods" `
  -FailureMessage "inference-http deployment not found or no ready pods"

# Test 2: Worker deployment exists
Test-ValidationStep -Name "Worker Deployment Running" -Test {
    $deployment = kubectl get deployment inference-worker --no-headers 2>$null
    $ready = ($deployment -split '\s+')[1]
    $ready -eq '1/1'
} -SuccessMessage "inference-worker deployment is running (1/1)" `
  -FailureMessage "inference-worker deployment not ready"

# Test 3: HPA exists and targets HTTP deployment
Test-ValidationStep -Name "HPA Configured" -Test {
    $hpa = kubectl get hpa inference-http-hpa --no-headers 2>$null
    $hpa -match 'inference-http'
} -SuccessMessage "HPA configured for inference-http" `
  -FailureMessage "HPA not found or misconfigured"

# Test 4: Service exists for HTTP
Test-ValidationStep -Name "HTTP Service Exists" -Test {
    $service = kubectl get service inference-http --no-headers 2>$null
    $null -ne $service
} -SuccessMessage "inference-http service created" `
  -FailureMessage "inference-http service not found"

Write-Host "`n=== CPU Usage Checks ===" -ForegroundColor Cyan

# Test 5: HTTP pods have lower CPU than old monolith
Test-ValidationStep -Name "HTTP Pod CPU Reduced" -Test {
    Start-Sleep -Seconds 15  # Wait for metrics
    $pods = kubectl top pods -l app=inference-http --no-headers 2>$null
    if (-not $pods) { return $false }
    
    $cpuValues = $pods | ForEach-Object {
        $parts = $_ -split '\s+'
        [int]($parts[1] -replace 'm','')
    }
    
    $avgCpu = ($cpuValues | Measure-Object -Average).Average
    Write-Host "  Average HTTP pod CPU: $([math]::Round($avgCpu))m" -ForegroundColor Gray
    
    # Success if average < 150m (old was ~238m)
    $avgCpu -lt 150
} -SuccessMessage "HTTP pods using significantly less CPU (< 150m avg)" `
  -FailureMessage "HTTP pods still using high CPU"

# Test 6: Worker pod has minimal CPU
Test-ValidationStep -Name "Worker Pod CPU Minimal" -Test {
    $pod = kubectl top pods -l app=inference-worker --no-headers 2>$null
    if (-not $pod) { return $false }
    
    $cpuValue = [int](($pod -split '\s+')[1] -replace 'm','')
    Write-Host "  Worker pod CPU: $($cpuValue)m" -ForegroundColor Gray
    
    # Success if < 200m
    $cpuValue -lt 200
} -SuccessMessage "Worker pod using minimal CPU (< 200m)" `
  -FailureMessage "Worker pod using excessive CPU"

Write-Host "`n=== Kafka Configuration Checks ===" -ForegroundColor Cyan

# Test 7: Worker has Kafka env vars, HTTP doesn't
Test-ValidationStep -Name "Kafka Env Separation" -Test {
    $workerEnv = kubectl get deployment inference-worker -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="KAFKA_BOOTSTRAP_SERVERS")].value}' 2>$null
    $httpEnv = kubectl get deployment inference-http -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="KAFKA_BOOTSTRAP_SERVERS")].value}' 2>$null
    
    Write-Host "  Worker has Kafka config: $($null -ne $workerEnv)" -ForegroundColor Gray
    Write-Host "  HTTP has Kafka config: $($null -ne $httpEnv)" -ForegroundColor Gray
    
    # Worker should have Kafka, HTTP should not
    ($null -ne $workerEnv) -and ($null -eq $httpEnv)
} -SuccessMessage "Kafka env vars correctly separated (worker: yes, HTTP: no)" `
  -FailureMessage "Kafka env vars not properly separated"

Write-Host "`n=== Functional Checks ===" -ForegroundColor Cyan

# Test 8: HTTP /healthz responds
Test-ValidationStep -Name "HTTP Healthz Endpoint" -Test {
    $pod = kubectl get pods -l app=inference-http --no-headers | Select-Object -First 1 | ForEach-Object { ($_ -split '\s+')[0] }
    if (-not $pod) { return $false }
    
    $health = kubectl exec $pod -- wget -qO- http://localhost:8000/healthz 2>$null
    $health -match 'status'
} -SuccessMessage "HTTP /healthz endpoint responding" `
  -FailureMessage "HTTP /healthz endpoint not responding"

# Test 9: Worker process running
Test-ValidationStep -Name "Worker Process Active" -Test {
    $pod = kubectl get pods -l app=inference-worker --no-headers | ForEach-Object { ($_ -split '\s+')[0] }
    if (-not $pod) { return $false }
    
    $process = kubectl exec $pod -- ps aux 2>$null | Select-String "inference_worker.py"
    $null -ne $process
} -SuccessMessage "Worker process (inference_worker.py) is running" `
  -FailureMessage "Worker process not found"

# Test 10: HPA can scale down
Test-ValidationStep -Name "HPA Scale-Down Capable" -Test {
    $hpa = kubectl get hpa inference-http-hpa -o json 2>$null | ConvertFrom-Json
    $currentReplicas = $hpa.status.currentReplicas
    $minReplicas = $hpa.spec.minReplicas
    
    Write-Host "  Current replicas: $currentReplicas, Min: $minReplicas" -ForegroundColor Gray
    
    # Success if current <= 3 or can scale down from current
    $currentReplicas -le 3 -or ($currentReplicas -gt $minReplicas)
} -SuccessMessage "HPA can scale down (current replicas reasonable)" `
  -FailureMessage "HPA stuck at high replica count"

Write-Host "`n=== Validation Summary ===" -ForegroundColor Cyan
Write-Host ""

$passCount = ($validationResults | Where-Object { $_.Status -eq "PASS" }).Count
$failCount = ($validationResults | Where-Object { $_.Status -eq "FAIL" }).Count
$totalCount = $validationResults.Count

$validationResults | Format-Table -AutoSize

Write-Host "`nResults: $passCount/$totalCount passed" -ForegroundColor $(if ($failCount -eq 0) { "Green" } else { "Yellow" })

if ($failCount -eq 0) {
    Write-Host "`n✓ All validation checks passed!" -ForegroundColor Green
    Write-Host "`nArchitecture successfully refactored:"
    Write-Host "  - HTTP-only inference pods (no Kafka overhead)"
    Write-Host "  - Dedicated Kafka worker (1 replica)"
    Write-Host "  - CPU-based HPA on HTTP pods"
    Write-Host "  - Reduced idle CPU usage"
    exit 0
} else {
    Write-Host "`n⚠ Some validation checks failed" -ForegroundColor Yellow
    Write-Host "Review failed tests above and check pod logs:"
    Write-Host "  kubectl logs -l app=inference-http"
    Write-Host "  kubectl logs -l app=inference-worker"
    exit 1
}
