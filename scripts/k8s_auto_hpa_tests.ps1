<#
.SYNOPSIS
    Kubernetes-based Distributed Locust HPA Testing Suite

.DESCRIPTION
    Automates large-scale Locust tests against Kubernetes deployments.
    Measures load-test metrics (RPS, latency, failures) and autoscaler metrics
    (replica count, CPU usage, scaling events).

.PARAMETER TestDuration
    Duration of each test in seconds (default: 120)

.PARAMETER InitialReplicas
    Starting replica count before HPA takes over (default: 2)

.PARAMETER WorkerCounts
    Array of Locust worker counts to test (default: @(4,8))

.PARAMETER UserCounts
    Array of concurrent user counts to test (default: @(50,100,200,400))

.PARAMETER HPAMinReplicas
    Minimum replicas for HPA (default: 2)

.PARAMETER HPAMaxReplicas
    Maximum replicas for HPA (default: 20)

.PARAMETER HPATargetCPU
    Target CPU utilization percentage for HPA (default: 70)

.EXAMPLE
    .\k8s_auto_hpa_tests.ps1
    
.EXAMPLE
    .\k8s_auto_hpa_tests.ps1 -TestDuration 180 -UserCounts @(100,200,400) -HPAMaxReplicas 15
#>

[CmdletBinding()]
param(
    [int]$TestDuration = 120,
    [int]$InitialReplicas = 2,
    [int[]]$WorkerCounts = @(4, 8),
    [int[]]$UserCounts = @(50, 100, 200, 400),
    [int]$HPAMinReplicas = 2,
    [int]$HPAMaxReplicas = 20,
    [int]$HPATargetCPU = 70,
    [string]$ResultsBaseDir = "reports/k8s_hpa_performance",
    [string]$InferenceDeployment = "inference",
    [string]$LocustMasterDeployment = "locust-master",
    [string]$LocustWorkerDeployment = "locust-worker",
    [int]$MonitoringInterval = 10
)

# ============================================================================
# COLOR OUTPUT FUNCTIONS
# ============================================================================

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message) Write-ColorOutput "[OK] $Message" "Green" }
function Write-Info { param([string]$Message) Write-ColorOutput "[INFO] $Message" "Cyan" }
function Write-Warning { param([string]$Message) Write-ColorOutput "[WARN] $Message" "Yellow" }
function Write-Error { param([string]$Message) Write-ColorOutput "[ERROR] $Message" "Red" }
function Write-Running { param([string]$Message) Write-ColorOutput "[RUN] $Message" "Yellow" }
function Write-Scale { param([string]$Message) Write-ColorOutput "[SCALE] $Message" "Magenta" }

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

$script:AllResults = @()
$script:SummaryFile = Join-Path $ResultsBaseDir "k8s_auto_summary.csv"
$script:SummaryMdFile = Join-Path $ResultsBaseDir "k8s_auto_summary.md"
$script:DetailedLog = Join-Path $ResultsBaseDir "detailed_metrics.log"

# ============================================================================
# KUBERNETES HELPER FUNCTIONS
# ============================================================================

function Test-KubernetesConnection {
    Write-Info "Testing Kubernetes connection..."
    try {
        $null = kubectl cluster-info 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Connected to Kubernetes cluster"
            return $true
        }
    } catch {}
    Write-Error "Cannot connect to Kubernetes cluster"
    return $false
}

function Get-HPAStatus {
    try {
        $hpaJson = kubectl get hpa $InferenceDeployment -o json 2>$null | ConvertFrom-Json
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
            
            return @{
                CurrentReplicas = $currentReplicas
                DesiredReplicas = $desiredReplicas
                CurrentCPU = $currentCPU
                Exists = $true
            }
        }
    } catch {}
    
    return @{
        CurrentReplicas = 0
        DesiredReplicas = 0
        CurrentCPU = 0
        Exists = $false
    }
}

function Ensure-HPA {
    param(
        [int]$MinReplicas = 2,
        [int]$MaxReplicas = 20,
        [int]$TargetCPU = 70
    )
    
    Write-Info "Checking HPA configuration for '$InferenceDeployment'..."
    
    $hpaStatus = Get-HPAStatus
    
    if ($hpaStatus.Exists) {
        # Get current HPA config
        try {
            $hpaJson = kubectl get hpa $InferenceDeployment -o json 2>$null | ConvertFrom-Json
            $currentMin = $hpaJson.spec.minReplicas
            $currentMax = $hpaJson.spec.maxReplicas
            $currentTarget = 0
            
            if ($hpaJson.spec.metrics) {
                foreach ($metric in $hpaJson.spec.metrics) {
                    if ($metric.type -eq "Resource" -and $metric.resource.name -eq "cpu") {
                        $currentTarget = $metric.resource.target.averageUtilization
                        break
                    }
                }
            }
            
            Write-Success "HPA exists: min=$currentMin, max=$currentMax, targetCPU=$currentTarget%"
            
            # Check if we need to update
            if ($currentMin -ne $MinReplicas -or $currentMax -ne $MaxReplicas -or $currentTarget -ne $TargetCPU) {
                Write-Info "Updating HPA to: min=$MinReplicas, max=$MaxReplicas, targetCPU=$TargetCPU%"
                
                # Patch the HPA
                $patchJson = @{
                    spec = @{
                        minReplicas = $MinReplicas
                        maxReplicas = $MaxReplicas
                        metrics = @(
                            @{
                                type = "Resource"
                                resource = @{
                                    name = "cpu"
                                    target = @{
                                        type = "Utilization"
                                        averageUtilization = $TargetCPU
                                    }
                                }
                            }
                        )
                    }
                } | ConvertTo-Json -Depth 10
                
                $patchJson | kubectl apply -f - 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "HPA updated successfully"
                    return $true
                } else {
                    # Try patching instead
                    kubectl patch hpa $InferenceDeployment --type merge -p "{`"spec`":{`"minReplicas`":$MinReplicas,`"maxReplicas`":$MaxReplicas}}" 2>&1 | Out-Null
                    if ($LASTEXITCODE -eq 0) {
                        Write-Success "HPA patched successfully"
                        return $true
                    }
                }
            } else {
                Write-Success "HPA configuration is already correct"
                return $true
            }
        } catch {
            Write-Warning "Could not verify HPA configuration: $_"
        }
    } else {
        Write-Info "Creating new HPA..."
        
        # Create HPA
        $hpaYaml = @"
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: $InferenceDeployment
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: $InferenceDeployment
  minReplicas: $MinReplicas
  maxReplicas: $MaxReplicas
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: $TargetCPU
"@
        
        $hpaYaml | kubectl apply -f - 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "HPA created successfully: min=$MinReplicas, max=$MaxReplicas, targetCPU=$TargetCPU%"
            return $true
        } else {
            Write-Error "Failed to create HPA"
            return $false
        }
    }
    
    return $true
}

function Get-InferenceReplicaCount {
    try {
        $deployment = kubectl get deployment $InferenceDeployment -o json 2>$null | ConvertFrom-Json
        if ($deployment) {
            return @{
                Desired = $deployment.spec.replicas
                Ready = $deployment.status.readyReplicas
                Available = $deployment.status.availableReplicas
            }
        }
    } catch {}
    
    return @{
        Desired = 0
        Ready = 0
        Available = 0
    }
}

function Set-InitialReplicas {
    param([int]$ReplicaCount)
    
    Write-Info "Setting initial replicas to $ReplicaCount (HPA will manage scaling from here)..."
    try {
        kubectl scale deployment $InferenceDeployment --replicas=$ReplicaCount 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Set initial replica count to $ReplicaCount"
            
            # Wait for pods to be ready
            Write-Info "Waiting for pods to become ready..."
            $timeout = 120
            $elapsed = 0
            while ($elapsed -lt $timeout) {
                $status = Get-InferenceReplicaCount
                if ($status.Ready -ge $ReplicaCount) {
                    Write-Success "Initial $ReplicaCount replicas are ready"
                    return $true
                }
                Start-Sleep -Seconds 3
                $elapsed += 3
            }
            Write-Warning "Timeout waiting for replicas to be ready"
            return $false
        }
    } catch {}
    Write-Error "Failed to set initial replicas"
    return $false
}

function Scale-LocustWorkers {
    param([int]$WorkerCount)
    
    Write-Info "Scaling Locust workers to $WorkerCount..."
    try {
        kubectl scale deployment $LocustWorkerDeployment --replicas=$WorkerCount 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Scaled Locust workers to $WorkerCount"
            Start-Sleep -Seconds 10  # Wait for workers to connect
            return $true
        }
    } catch {}
    Write-Error "Failed to scale Locust workers"
    return $false
}

function Wait-ForLocustMaster {
    Write-Info "Waiting for Locust master to be ready..."
    $maxAttempts = 30
    for ($i = 1; $i -le $maxAttempts; $i++) {
        try {
            $pods = kubectl get pod -l app=locust,role=master -o json 2>$null | ConvertFrom-Json
            if ($pods.items -and $pods.items.Count -gt 0) {
                $pod = $pods.items[0]
                $podName = $pod.metadata.name
                
                # Check if pod is ready
                $readyCondition = $pod.status.conditions | Where-Object { $_.type -eq "Ready" } | Select-Object -First 1
                if ($readyCondition -and $readyCondition.status -eq "True") {
                    Write-Success "Locust master is ready: $podName"
                    return $podName
                }
            }
        } catch {
            # Silently retry
        }
        Start-Sleep -Seconds 2
    }
    Write-Error "Locust master not ready after $($maxAttempts * 2) seconds"
    return $null
}

# ============================================================================
# LOCUST TEST FUNCTIONS
# ============================================================================

function Invoke-LocustTest {
    param(
        [int]$Users,
        [int]$SpawnRate,
        [int]$Duration,
        [string]$MasterPod,
        [int]$ExpectedWorkers = 4
    )
    
    # Use CSV output for reliable stats collection with distributed workers
    $csvPrefix = "/tmp/locust_test_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    $cmd = "cd /home/locust && locust -f locustfile.py --headless --host=http://inference:8000 -u $Users -r $SpawnRate -t ${Duration}s --expect-workers=$ExpectedWorkers --csv=$csvPrefix"
    
    Write-Running "Starting Locust test: $Users users, ${SpawnRate}/s spawn rate, ${Duration}s duration"
    Write-Info "Expecting $ExpectedWorkers workers to connect..."
    
    try {
        # Use explicit flags to avoid hanging on Windows
        $result = & kubectl exec --stdin=false --tty=false deployment/$LocustMasterDeployment -- sh -c $cmd 2>&1
        $output = $result | Out-String
        
        # Check if test completed
        if ($output -match "Shutting down|All users spawned") {
            # Read CSV stats file for accurate metrics
            $csvStatsFile = "${csvPrefix}_stats.csv"
            Write-Info "Retrieving stats from CSV: $csvStatsFile"
            
            $csvContent = kubectl exec --stdin=false --tty=false deployment/$LocustMasterDeployment -- cat $csvStatsFile 2>$null
            
            if ($csvContent) {
                # Clean up CSV files
                kubectl exec --stdin=false --tty=false deployment/$LocustMasterDeployment -- rm -f "${csvPrefix}*.csv" 2>$null | Out-Null
                return @{
                    Output = $output
                    CSV = $csvContent | Out-String
                }
            } else {
                Write-Warning "Could not retrieve CSV stats, falling back to terminal output"
                return @{
                    Output = $output
                    CSV = ""
                }
            }
        } else {
            Write-Warning "Test may have been interrupted or did not complete normally"
            return @{
                Output = $output
                CSV = ""
            }
        }
    } catch {
        Write-Error "Failed to execute Locust test: $_"
        return @{
            Output = ""
            CSV = ""
        }
    }
}

function Parse-LocustOutput {
    param(
        [string]$Output,
        [string]$CSVContent = ""
    )
    
    $metrics = @{
        TotalRequests = 0
        Failures = 0
        AvgLatency = 0
        P50Latency = 0
        P95Latency = 0
        P99Latency = 0
        Throughput = 0.0
        FailureRate = 0.0
    }
    
    # Try CSV parsing first (most reliable for distributed workers)
    if ($CSVContent) {
        Write-Info "Parsing CSV stats..."
        
        # Clean up CSV content - remove any carriage returns and normalize line breaks
        $cleanedCSV = $CSVContent -replace "`r", "" -replace "`n`n+", "`n"
        $lines = $cleanedCSV -split "`n"
        
        # CSV format: Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%
        foreach ($line in $lines) {
            # Look for line starting with comma (wrapped Aggregated line) or "Aggregated"
            if ($line -match '^,?Aggregated' -or $line -match '^"?Aggregated') {
                $fields = $line -split ','
                
                # When line starts with comma: [0]=empty, [1]=Aggregated, [2]=RequestCount
                # When line starts normally: [0]=Type, [1]=Aggregated, [2]=RequestCount
                # Both cases have Request Count at index 2
                
                if ($fields.Count -ge 20) {
                    $metrics.TotalRequests = [int](($fields[2] -replace '"','').Trim())
                    $metrics.Failures = [int](($fields[3] -replace '"','').Trim())
                    $metrics.P50Latency = [int][decimal](($fields[4] -replace '"','').Trim())
                    $metrics.AvgLatency = [int][decimal](($fields[5] -replace '"','').Trim())
                    $metrics.P95Latency = [int][decimal](($fields[16] -replace '"','').Trim())
                    $metrics.P99Latency = [int][decimal](($fields[18] -replace '"','').Trim())
                    $metrics.Throughput = [decimal](($fields[9] -replace '"','').Trim())
                    
                    if ($metrics.TotalRequests -gt 0) {
                        $metrics.FailureRate = [math]::Round(($metrics.Failures / $metrics.TotalRequests) * 100, 2)
                    }
                    
                    Write-Success "CSV parsing successful: $($metrics.TotalRequests) requests"
                    return $metrics
                }
            }
        }
        Write-Warning "Could not find Aggregated stats in CSV"
    }
    
    # Fallback to terminal output parsing
    Write-Info "Parsing terminal output..."
    
    # Parse final Aggregated stats line from stats table
    # Format: "Aggregated   159   0(0.00%) |  147  73  559  120 |  5.60   0.00"
    if ($Output -match 'Aggregated\s+(\d+)\s+(\d+)\([^\)]+\)\s+\|\s+(\d+)\s+\d+\s+\d+\s+(\d+)\s+\|\s+([\d\.]+)') {
        $metrics.TotalRequests = [int]$Matches[1]
        $metrics.Failures = [int]$Matches[2]
        $metrics.AvgLatency = [int]$Matches[3]
        $metrics.P50Latency = [int]$Matches[4]  # Median
        $metrics.Throughput = [decimal]$Matches[5]
        
        if ($metrics.TotalRequests -gt 0) {
            $metrics.FailureRate = [math]::Round(($metrics.Failures / $metrics.TotalRequests) * 100, 2)
        }
    }
    
    # Parse percentiles table for P95/P99
    # Look for the last occurrence before "Type     Name" pattern
    $lines = $Output -split "`n"
    $percentileLines = $lines | Where-Object { $_ -match '^\s*Aggregated\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)' }
    if ($percentileLines) {
        $lastPercentileLine = $percentileLines[-1]
        if ($lastPercentileLine -match '^\s*Aggregated\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+\d+\s+(\d+)') {
            $metrics.P95Latency = [int]$Matches[1]
            $metrics.P99Latency = [int]$Matches[2]
        }
    }
    
    if ($metrics.TotalRequests -eq 0) {
        Write-Warning "No requests found in output - workers may not have connected properly"
    } else {
        Write-Success "Terminal parsing successful: $($metrics.TotalRequests) requests"
    }
    
    return $metrics
}

# ============================================================================
# MONITORING FUNCTIONS
# ============================================================================

function Start-MetricsMonitoring {
    param(
        [int]$Duration,
        [string]$TestName
    )
    
    $metricsLog = @()
    $startTime = Get-Date
    $endTime = $startTime.AddSeconds($Duration)
    $lastReplicaCount = 0
    
    Write-Info "Starting metrics monitoring for ${Duration}s..."
    
    while ((Get-Date) -lt $endTime) {
        $elapsed = [int]((Get-Date) - $startTime).TotalSeconds
        
        # Get HPA status
        $hpa = Get-HPAStatus
        $replicas = Get-InferenceReplicaCount
        
        # Detect scaling events
        if ($replicas.Ready -ne $lastReplicaCount -and $lastReplicaCount -gt 0) {
            if ($replicas.Ready -gt $lastReplicaCount) {
                Write-Scale "SCALE-UP → $($replicas.Ready) replicas at ${elapsed}s"
            } else {
                Write-Scale "SCALE-DOWN → $($replicas.Ready) replicas at ${elapsed}s"
            }
        }
        $lastReplicaCount = $replicas.Ready
        
        # Log metrics
        $snapshot = @{
            Timestamp = Get-Date -Format "HH:mm:ss"
            Elapsed = $elapsed
            CurrentReplicas = $replicas.Ready
            DesiredReplicas = $hpa.DesiredReplicas
            CurrentCPU = $hpa.CurrentCPU
        }
        $metricsLog += $snapshot
        
        # Live display
        $progress = [math]::Min([math]::Round(($elapsed / $Duration) * 100), 100)
        $progressBar = "█" * [math]::Floor($progress / 5) + "░" * (20 - [math]::Floor($progress / 5))
        
        Write-Host "`r[$progressBar] $progress% | Replicas: $($replicas.Ready)/$($hpa.DesiredReplicas) | CPU: $($hpa.CurrentCPU)% | ${elapsed}s/$Duration" -NoNewline -ForegroundColor Cyan
        
        Start-Sleep -Seconds $MonitoringInterval
    }
    
    Write-Host ""  # Newline after progress
    return $metricsLog
}

function Get-MetricsSummary {
    param([array]$MetricsLog)
    
    if (-not $MetricsLog -or $MetricsLog.Count -eq 0) {
        return @{
            InitialReplicas = 0
            PeakReplicas = 0
            FinalReplicas = 0
            MinReplicas = 0
            MaxReplicas = 0
            AvgCPU = 0
            ScalingEvents = 0
            ScaleUpEvents = 0
            ScaleDownEvents = 0
        }
    }
    
    # Handle both array of hashtables and array of strings/null
    $validMetrics = $MetricsLog | Where-Object { $_ -and $_.CurrentReplicas }
    
    if ($validMetrics.Count -eq 0) {
        return @{
            InitialReplicas = 0
            PeakReplicas = 0
            FinalReplicas = 0
            MinReplicas = 0
            MaxReplicas = 0
            AvgCPU = 0
            ScalingEvents = 0
            ScaleUpEvents = 0
            ScaleDownEvents = 0
        }
    }
    
    # Calculate CPU average
    $cpuValues = $validMetrics | Where-Object { $_.CurrentCPU -and $_.CurrentCPU -gt 0 } | ForEach-Object { $_.CurrentCPU }
    $avgCPU = if ($cpuValues) { [math]::Round(($cpuValues | Measure-Object -Average).Average, 1) } else { 0 }
    
    # Calculate replica statistics
    $replicaCounts = $validMetrics | ForEach-Object { $_.CurrentReplicas }
    $initialReplicas = $validMetrics[0].CurrentReplicas
    $finalReplicas = $validMetrics[-1].CurrentReplicas
    $minReplicas = ($replicaCounts | Measure-Object -Minimum).Minimum
    $maxReplicas = ($replicaCounts | Measure-Object -Maximum).Maximum
    $peakReplicas = $maxReplicas
    
    # Count scaling events (replica count changes)
    $scalingEvents = 0
    $scaleUpEvents = 0
    $scaleDownEvents = 0
    
    for ($i = 1; $i -lt $validMetrics.Count; $i++) {
        $current = $validMetrics[$i].CurrentReplicas
        $previous = $validMetrics[$i-1].CurrentReplicas
        
        if ($current -ne $previous) {
            $scalingEvents++
            if ($current -gt $previous) {
                $scaleUpEvents++
            } else {
                $scaleDownEvents++
            }
        }
    }
    
    return @{
        InitialReplicas = $initialReplicas
        PeakReplicas = $peakReplicas
        FinalReplicas = $finalReplicas
        MinReplicas = $minReplicas
        MaxReplicas = $maxReplicas
        AvgCPU = $avgCPU
        ScalingEvents = $scalingEvents
        ScaleUpEvents = $scaleUpEvents
        ScaleDownEvents = $scaleDownEvents
    }
}

# ============================================================================
# TEST EXECUTION
# ============================================================================

function Run-SingleTest {
    param(
        [int]$Workers,
        [int]$Users,
        [int]$Duration,
        [string]$MasterPod
    )
    
    $testName = "workers${Workers}_u${Users}_hpa"
    
    Write-Host "`n" + ("=" * 100) -ForegroundColor Cyan
    Write-ColorOutput "TEST: $testName" "Magenta"
    Write-ColorOutput "  Locust Workers: $Workers" "White"
    Write-ColorOutput "  Concurrent Users: $Users" "White"
    Write-ColorOutput "  Duration: ${Duration}s" "White"
    Write-ColorOutput "  HPA: Enabled (dynamic scaling)" "Green"
    Write-Host ("=" * 100) -ForegroundColor Cyan
    
    # Get current HPA configuration
    $hpaStatus = Get-HPAStatus
    if ($hpaStatus.Exists) {
        try {
            $hpaJson = kubectl get hpa $InferenceDeployment -o json 2>$null | ConvertFrom-Json
            $hpaMin = $hpaJson.spec.minReplicas
            $hpaMax = $hpaJson.spec.maxReplicas
            $hpaTarget = 0
            
            if ($hpaJson.spec.metrics) {
                foreach ($metric in $hpaJson.spec.metrics) {
                    if ($metric.type -eq "Resource" -and $metric.resource.name -eq "cpu") {
                        $hpaTarget = $metric.resource.target.averageUtilization
                        break
                    }
                }
            }
            
            Write-Info "HPA Configuration: min=$hpaMin, max=$hpaMax, targetCPU=$hpaTarget%"
            Write-Info "Current state: $($hpaStatus.CurrentReplicas) replicas, CPU=$($hpaStatus.CurrentCPU)%"
        } catch {
            Write-Warning "Could not read HPA details"
        }
    } else {
        Write-Warning "HPA not found - scaling may not work"
    }
    
    # Calculate spawn rate
    $spawnRate = [math]::Max([math]::Floor($Users / 15), 5)
    
    # Start monitoring in background
    Write-Info "Starting test and HPA metrics monitoring..."
    
    # Record initial state
    $initialReplicas = $hpaStatus.CurrentReplicas
    if ($initialReplicas -eq 0) {
        $deployStatus = Get-InferenceReplicaCount
        $initialReplicas = $deployStatus.Ready
    }
    
    # Start monitoring job
    $monitoringJob = Start-Job -ScriptBlock {
        param($Duration, $Interval, $Deployment)
        $metrics = @()
        $start = Get-Date
        while (((Get-Date) - $start).TotalSeconds -lt ($Duration + 10)) {  # Extra 10s for data
            try {
                $hpaJson = kubectl get hpa $Deployment -o json 2>$null | ConvertFrom-Json
                $depJson = kubectl get deployment $Deployment -o json 2>$null | ConvertFrom-Json
                
                $currentCPU = 0
                if ($hpaJson.status.currentMetrics) {
                    foreach ($metric in $hpaJson.status.currentMetrics) {
                        if ($metric.type -eq "Resource" -and $metric.resource.name -eq "cpu") {
                            $currentCPU = $metric.resource.current.averageUtilization
                            break
                        }
                    }
                }
                
                $metrics += @{
                    Timestamp = Get-Date -Format "HH:mm:ss"
                    Elapsed = [int]((Get-Date) - $start).TotalSeconds
                    CurrentReplicas = [int]($depJson.status.readyReplicas)
                    DesiredReplicas = [int]($hpaJson.status.desiredReplicas)
                    CurrentCPU = [int]$currentCPU
                }
            } catch {}
            Start-Sleep -Seconds $Interval
        }
        return $metrics
    } -ArgumentList $Duration, $MonitoringInterval, $InferenceDeployment
    
    # Get current worker count for Locust
    try {
        $workerPods = kubectl get pods -l app=locust,role=worker -o json 2>$null | ConvertFrom-Json
        $actualWorkers = $workerPods.items.Count
        Write-Info "Detected $actualWorkers Locust workers"
    } catch {
        $actualWorkers = $Workers
        Write-Warning "Could not detect worker count, using expected: $Workers"
    }
    
    # Run Locust test with worker count
    $testResult = Invoke-LocustTest -Users $Users -SpawnRate $spawnRate -Duration $Duration -MasterPod $MasterPod -ExpectedWorkers $actualWorkers
    
    # Wait for monitoring to complete
    Write-Info "Collecting final HPA metrics..."
    $metricsLog = Receive-Job -Job $monitoringJob -Wait
    Remove-Job -Job $monitoringJob
    
    if (-not $testResult.Output) {
        Write-Error "No output from Locust test"
        return $null
    }
    
    # Parse Locust metrics (CSV first, then fallback to terminal output)
    $locustMetrics = Parse-LocustOutput -Output $testResult.Output -CSVContent $testResult.CSV
    
    # Get autoscaling summary
    $hpaSummary = Get-MetricsSummary -MetricsLog $metricsLog
    
    # Override initial replicas if we recorded it
    if ($initialReplicas -gt 0) {
        $hpaSummary.InitialReplicas = $initialReplicas
    }
    
    # Display results
    Write-Host "`n[RESULTS] Test Results:" -ForegroundColor Green
    Write-Host "  Locust Metrics:" -ForegroundColor Yellow
    Write-Host "    Total Requests:  $($locustMetrics.TotalRequests)" -ForegroundColor White
    Write-Host "    Failures:        $($locustMetrics.Failures) ($($locustMetrics.FailureRate)%)" -ForegroundColor $(if ($locustMetrics.FailureRate -gt 0) { "Red" } else { "Green" })
    Write-Host "    RPS:             $($locustMetrics.Throughput)" -ForegroundColor White
    Write-Host "    Avg Latency:     $($locustMetrics.AvgLatency)ms" -ForegroundColor White
    Write-Host "    P50 Latency:     $($locustMetrics.P50Latency)ms" -ForegroundColor White
    Write-Host "    P95 Latency:     $($locustMetrics.P95Latency)ms" -ForegroundColor White
    Write-Host "    P99 Latency:     $($locustMetrics.P99Latency)ms" -ForegroundColor White
    
    Write-Host "  HPA Metrics:" -ForegroundColor Yellow
    Write-Host "    Min/Max Replicas:  $($hpaSummary.MinReplicas)/$($hpaSummary.MaxReplicas)" -ForegroundColor White
    Write-Host "    Initial Replicas:  $($hpaSummary.InitialReplicas)" -ForegroundColor White
    Write-Host "    Peak Replicas:     $($hpaSummary.PeakReplicas)" -ForegroundColor $(if ($hpaSummary.PeakReplicas -gt $hpaSummary.InitialReplicas) { "Cyan" } else { "White" })
    Write-Host "    Final Replicas:    $($hpaSummary.FinalReplicas)" -ForegroundColor White
    Write-Host "    Avg CPU:           $($hpaSummary.AvgCPU)%" -ForegroundColor $(if ($hpaSummary.AvgCPU -gt 70) { "Yellow" } else { "White" })
    Write-Host "    Scaling Events:    $($hpaSummary.ScalingEvents) ($($hpaSummary.ScaleUpEvents) scale-up, $($hpaSummary.ScaleDownEvents) scale-down)" -ForegroundColor $(if ($hpaSummary.ScalingEvents -gt 0) { "Magenta" } else { "White" })
    
    # Create result object
    $result = [PSCustomObject]@{
        Workers = $Workers
        Users = $Users
        InitialReplicas = $hpaSummary.InitialReplicas
        PeakReplicas = $hpaSummary.PeakReplicas
        FinalReplicas = $hpaSummary.FinalReplicas
        MinReplicas = $hpaSummary.MinReplicas
        MaxReplicas = $hpaSummary.MaxReplicas
        AvgCPU = $hpaSummary.AvgCPU
        ScalingEvents = $hpaSummary.ScalingEvents
        ScaleUpEvents = $hpaSummary.ScaleUpEvents
        ScaleDownEvents = $hpaSummary.ScaleDownEvents
        RPS = $locustMetrics.Throughput
        Median_ms = $locustMetrics.P50Latency
        P95_ms = $locustMetrics.P95Latency
        P99_ms = $locustMetrics.P99Latency
        AvgLatency_ms = $locustMetrics.AvgLatency
        Failures = $locustMetrics.Failures
        Failures_Pct = $locustMetrics.FailureRate
        TotalRequests = $locustMetrics.TotalRequests
        Duration_s = $Duration
    }
    
    Write-Success "Test completed: $testName"
    return $result
}

# ============================================================================
# INITIALIZATION AND CLEANUP
# ============================================================================

function Initialize-TestEnvironment {
    Write-Info "Initializing test environment..."
    
    # Create results directory
    New-Item -ItemType Directory -Force -Path $ResultsBaseDir | Out-Null
    
    # Initialize CSV with headers
    "Workers,Users,InitialReplicas,PeakReplicas,FinalReplicas,MinReplicas,MaxReplicas,AvgCPU%,ScalingEvents,ScaleUpEvents,ScaleDownEvents,RPS,Median_ms,P95_ms,P99_ms,AvgLatency_ms,Failures,Failures_Pct,TotalRequests,Duration_s" | 
        Out-File -FilePath $script:SummaryFile -Encoding UTF8
    
    # Initialize detailed log
    "=== K8s HPA Performance Testing - Detailed Log ===" | Out-File -FilePath $script:DetailedLog -Encoding UTF8
    "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $script:DetailedLog -Append -Encoding UTF8
    "" | Out-File -FilePath $script:DetailedLog -Append -Encoding UTF8
    
    Write-Success "Results directory: $ResultsBaseDir"
}

function Export-MarkdownReport {
    param([array]$Results)
    
    $markdown = @"
# Kubernetes HPA Performance Testing Results

**Test Date**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")  
**Test Duration**: ${TestDuration}s per configuration  
**Total Tests**: $($Results.Count)  
**HPA Configuration**: min=$HPAMinReplicas, max=$HPAMaxReplicas, targetCPU=$HPATargetCPU%

## Configuration Matrix

- **Initial Replicas**: $InitialReplicas (baseline before HPA takes over)
- **Locust Workers**: $($WorkerCounts -join ', ')
- **User Counts**: $($UserCounts -join ', ')
- **Monitoring Interval**: ${MonitoringInterval}s

## Complete Results

| Workers | Users | Init→Peak→Final | Min/Max | Avg CPU% | Scaling Events | RPS | P50(ms) | P95(ms) | P99(ms) | Failures% |
|---------|-------|-----------------|---------|----------|----------------|-----|---------|---------|---------|-----------|
"@

    foreach ($result in $Results | Sort-Object Workers, Users) {
        $replicaRange = "$($result.InitialReplicas)->$($result.PeakReplicas)->$($result.FinalReplicas)"
        $minMax = "$($result.MinReplicas)/$($result.MaxReplicas)"
        $scalingInfo = "$($result.ScalingEvents) ($($result.ScaleUpEvents)up $($result.ScaleDownEvents)down)"
        $markdown += "`n| $($result.Workers) | $($result.Users) | $replicaRange | $minMax | $($result.AvgCPU) | $scalingInfo | $($result.RPS) | $($result.Median_ms) | $($result.P95_ms) | $($result.P99_ms) | $($result.Failures_Pct) |"
    }
    
    $markdown += @"

## Performance Analysis

### Throughput by Configuration

"@
    
    foreach ($workers in $WorkerCounts) {
        $markdown += "`n#### $workers Locust Workers`n`n"
        $markdown += "| Users | Init->Peak->Final | Avg CPU% | RPS | P95 Latency (ms) | Scaling Events |`n"
        $markdown += "|-------|-------------------|----------|-----|------------------|----------------|`n"
        
        foreach ($result in $Results | Where-Object { $_.Workers -eq $workers } | Sort-Object Users) {
            $replicaInfo = "$($result.InitialReplicas)->$($result.PeakReplicas)->$($result.FinalReplicas)"
            $markdown += "| $($result.Users) | $replicaInfo | $($result.AvgCPU)% | $($result.RPS) | $($result.P95_ms) | $($result.ScalingEvents) |`n"
        }
    }
    
    $markdown += @"

### HPA Scaling Analysis

**Tests with Autoscaling Activity:**

"@
    
    $scaledTests = $Results | Where-Object { $_.ScalingEvents -gt 0 } | Sort-Object -Descending ScalingEvents
    if ($scaledTests.Count -gt 0) {
        foreach ($test in $scaledTests) {
            $markdown += "- **$($test.Users) users, $($test.Workers) workers**: $($test.InitialReplicas) → $($test.FinalReplicas) replicas ($($test.ScalingEvents) events, Avg CPU: $($test.AvgCPU)%)`n"
        }
    } else {
        $markdown += "`nNo autoscaling events detected across all tests.`n"
    }
    
    $markdown += @"

### Key Findings

1. **Maximum Throughput**: $(($Results | Sort-Object RPS -Descending | Select-Object -First 1).RPS) req/s
2. **Best P95 Latency**: $(($Results | Sort-Object P95_ms | Select-Object -First 1).P95_ms)ms
3. **Most Scaling Events**: $(($Results | Sort-Object ScalingEvents -Descending | Select-Object -First 1).ScalingEvents) (at $($($Results | Sort-Object ScalingEvents -Descending | Select-Object -First 1).Users) users)
4. **CPU Efficiency**: Average $([math]::Round(($Results | Measure-Object -Property AvgCPU -Average).Average, 1))% across all tests
5. **Failure Rate**: $(if (($Results | Where-Object { $_.Failures_Pct -gt 0 }).Count -eq 0) { "0% across all tests (PASS)" } else { "$((($Results | Where-Object { $_.Failures_Pct -gt 0 }).Count)) tests had failures (WARN)" })
6. **Total Requests Processed**: $(($Results | Measure-Object -Property TotalRequests -Sum).Sum)

### Optimal Configurations

**Highest Throughput:**
"@
    
    $topRPS = $Results | Sort-Object RPS -Descending | Select-Object -First 3
    foreach ($result in $topRPS) {
        $markdown += "`n- **$($result.RPS) req/s** @ $($result.Users) users, $($result.Workers) workers, $($result.FinalReplicas) replicas"
    }
    
    $markdown += "`n`n**Lowest Latency (P95):**`n"
    $topLatency = $Results | Sort-Object P95_ms | Select-Object -First 3
    foreach ($result in $topLatency) {
        $markdown += "`n- **$($result.P95_ms)ms** @ $($result.Users) users, $($result.Workers) workers, $($result.FinalReplicas) replicas"
    }
    
    $markdown += @"


---

*Generated by k8s_auto_hpa_tests.ps1*
*Kubernetes Deployment: $InferenceDeployment*
*Locust Master: $LocustMasterDeployment*
"@
    
    $markdown | Out-File -FilePath $script:SummaryMdFile -Encoding UTF8
    Write-Success "Markdown report saved: $script:SummaryMdFile"
}

function Show-FinalSummary {
    param([array]$Results)
    
    Write-Host "`n`n" + ("=" * 120) -ForegroundColor Green
    Write-ColorOutput "═══ KUBERNETES HPA PERFORMANCE TESTING COMPLETE ═══" "Green"
    Write-Host ("=" * 120) -ForegroundColor Green
    
    Write-Host "`n[SUMMARY] Horizontal Scaling Performance Summary`n" -ForegroundColor Cyan
    
    # Table header
    $header = "{0,-7} {1,-8} {2,-10} {3,-8} {4,-13} {5,-8} {6,-10}" -f `
        "Users", "Workers", "Replicas", "AvgCPU%", "MaxReplicas", "RPS", "P95(ms)"
    Write-Host $header -ForegroundColor Yellow
    Write-Host ("-" * 120) -ForegroundColor White
    
    foreach ($result in $Results | Sort-Object Users, Workers) {
        $replicaRange = "$($result.InitialReplicas)→$($result.FinalReplicas)"
        $color = if ($result.Failures_Pct -gt 0) { "Red" } 
                 elseif ($result.ScalingEvents -gt 0) { "Magenta" }
                 elseif ($result.RPS -gt 50) { "Green" } 
                 else { "White" }
        
        $row = "{0,-7} {1,-8} {2,-10} {3,-8} {4,-13} {5,-8} {6,-10}" -f `
            $result.Users, $result.Workers, $replicaRange, "$($result.AvgCPU)%", 
            $result.MaxReplicas, $result.RPS, $result.P95_ms
        Write-Host $row -ForegroundColor $color
    }
    
    Write-Host ("-" * 120) -ForegroundColor White
    
    # Statistics
    $scalingTests = $Results | Where-Object { $_.ScalingEvents -gt 0 }
    $totalScalingEvents = ($Results | Measure-Object -Property ScalingEvents -Sum).Sum
    $totalScaleUp = ($Results | Measure-Object -Property ScaleUpEvents -Sum).Sum
    $totalScaleDown = ($Results | Measure-Object -Property ScaleDownEvents -Sum).Sum
    
    Write-Host "`n[METRICS] Key Performance Indicators:" -ForegroundColor Cyan
    Write-Host "  Total Tests Executed:     $($Results.Count)" -ForegroundColor White
    Write-Host "  Total Requests:           $(($Results | Measure-Object -Property TotalRequests -Sum).Sum)" -ForegroundColor White
    Write-Host "  Maximum Throughput:       $(($Results | Sort-Object RPS -Descending | Select-Object -First 1).RPS) req/s" -ForegroundColor Green
    Write-Host "  Best P95 Latency:         $(($Results | Sort-Object P95_ms | Select-Object -First 1).P95_ms)ms" -ForegroundColor Green
    Write-Host "  Average CPU Utilization:  $([math]::Round(($Results | Measure-Object -Property AvgCPU -Average).Average, 1))%" -ForegroundColor White
    
    Write-Host "`n[AUTOSCALING] HPA Activity:" -ForegroundColor Cyan
    Write-Host "  Tests with Scaling:       $($scalingTests.Count) / $($Results.Count)" -ForegroundColor $(if ($scalingTests.Count -gt 0) { "Magenta" } else { "White" })
    Write-Host "  Total Scaling Events:     $totalScalingEvents" -ForegroundColor $(if ($totalScalingEvents -gt 0) { "Magenta" } else { "White" })
    
    if ($totalScalingEvents -gt 0) {
        Write-Host "  Scale-Up Events:          $totalScaleUp" -ForegroundColor Green
        Write-Host "  Scale-Down Events:        $totalScaleDown" -ForegroundColor Yellow
    }
    
    $failedTests = $Results | Where-Object { $_.Failures_Pct -gt 0 }
    Write-Host "`n[RELIABILITY] Error Analysis:" -ForegroundColor Cyan
    if ($failedTests.Count -eq 0) {
        Write-Host "  Status:                   ALL TESTS PASSED (0 failures)" -ForegroundColor Green
    } else {
        Write-Host "  Failed Tests:             $($failedTests.Count) / $($Results.Count)" -ForegroundColor Red
        Write-Host "  Total Failures:           $(($failedTests | Measure-Object -Property Failures -Sum).Sum)" -ForegroundColor Red
    }
    
    Write-Host "`n[OUTPUT] Results Location:" -ForegroundColor Cyan
    Write-Host "  Directory:                $ResultsBaseDir" -ForegroundColor White
    Write-Host "  CSV Summary:              $script:SummaryFile" -ForegroundColor White
    Write-Host "  Markdown Report:          $script:SummaryMdFile" -ForegroundColor White
    Write-Host "  Detailed Log:             $script:DetailedLog" -ForegroundColor White
    
    Write-Host "`n" + ("=" * 120) -ForegroundColor Green
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function Main {
    Clear-Host
    
    Write-Host @"
╔════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                    ║
║           KUBERNETES HPA DISTRIBUTED LOCUST PERFORMANCE TESTING SUITE              ║
║                                                                                    ║
╚════════════════════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

    Write-Host "`n[CONFIGURATION] Test Parameters:" -ForegroundColor Yellow
    Write-Host "  Test Duration:            ${TestDuration}s per configuration" -ForegroundColor White
    Write-Host "  Initial Replicas:         $InitialReplicas (baseline)" -ForegroundColor White
    Write-Host "  HPA Min/Max Replicas:     $HPAMinReplicas / $HPAMaxReplicas" -ForegroundColor Green
    Write-Host "  HPA Target CPU:           $HPATargetCPU%" -ForegroundColor Green
    Write-Host "  Locust Worker Counts:     $($WorkerCounts -join ', ')" -ForegroundColor White
    Write-Host "  User Counts:              $($UserCounts -join ', ')" -ForegroundColor White
    Write-Host "  Monitoring Interval:      ${MonitoringInterval}s" -ForegroundColor White
    Write-Host "  Total Tests Planned:      $(($WorkerCounts.Count * $UserCounts.Count))" -ForegroundColor Green
    Write-Host "  Inference Deployment:     $InferenceDeployment" -ForegroundColor White
    Write-Host "  Locust Master:            $LocustMasterDeployment" -ForegroundColor White
    Write-Host "  Locust Worker:            $LocustWorkerDeployment" -ForegroundColor White
    
    # Pre-flight checks
    Write-Host "`n[PREFLIGHT] Running checks..." -ForegroundColor Yellow
    
    if (-not (Test-KubernetesConnection)) {
        Write-Error "Kubernetes connection failed. Exiting."
        return
    }
    
    # Initialize environment
    Initialize-TestEnvironment
    
    # Wait for Locust master
    $masterPod = Wait-ForLocustMaster
    if (-not $masterPod) {
        Write-Error "Locust master not available. Exiting."
        return
    }
    
    # Ensure HPA is configured correctly
    Write-Info "Configuring HPA..."
    if (-not (Ensure-HPA -MinReplicas $HPAMinReplicas -MaxReplicas $HPAMaxReplicas -TargetCPU $HPATargetCPU)) {
        Write-Error "Failed to configure HPA. Exiting."
        return
    }
    
    # Set initial replica count
    Write-Info "Setting initial replica baseline..."
    if (-not (Set-InitialReplicas -ReplicaCount $InitialReplicas)) {
        Write-Warning "Failed to set initial replicas, but continuing..."
    }
    Start-Sleep -Seconds 10
    
    # Main test loop
    Write-Host "`n" + ("=" * 120) -ForegroundColor Magenta
    Write-ColorOutput "[START] Beginning test execution..." "Magenta"
    Write-Host ("=" * 120) -ForegroundColor Magenta
    
    $testCounter = 0
    $totalTests = $WorkerCounts.Count * $UserCounts.Count
    
    foreach ($workers in $WorkerCounts) {
        Write-Host "`n" + ("=" * 120) -ForegroundColor Cyan
        Write-ColorOutput "[CONFIG] Switching to $workers Locust Workers" "Cyan"
        Write-Host ("=" * 120) -ForegroundColor Cyan
        
        if (-not (Scale-LocustWorkers -WorkerCount $workers)) {
            Write-Warning "Failed to scale to $workers workers, skipping this tier"
            continue
        }
        
        foreach ($users in $UserCounts) {
            $testCounter++
            Write-Info "═══ Progress: Test $testCounter of $totalTests ═══"
            
            $result = Run-SingleTest -Workers $workers -Users $users -Duration $TestDuration -MasterPod $masterPod
            
            if ($result) {
                $script:AllResults += $result
                
                # Append to CSV
                "$($result.Workers),$($result.Users),$($result.InitialReplicas),$($result.PeakReplicas),$($result.FinalReplicas),$($result.MinReplicas),$($result.MaxReplicas),$($result.AvgCPU),$($result.ScalingEvents),$($result.ScaleUpEvents),$($result.ScaleDownEvents),$($result.RPS),$($result.Median_ms),$($result.P95_ms),$($result.P99_ms),$($result.AvgLatency_ms),$($result.Failures),$($result.Failures_Pct),$($result.TotalRequests),$($result.Duration_s)" | 
                    Out-File -FilePath $script:SummaryFile -Append -Encoding UTF8
            }
            
            # Cool down between tests to let HPA stabilize
            if ($testCounter -lt $totalTests) {
                Write-Info "Cooling down for 30 seconds before next test (HPA stabilization)..."
                Start-Sleep -Seconds 30
            }
        }
    }
    
    # Generate final outputs
    Write-Host "`n[FINALIZATION] Generating reports..." -ForegroundColor Yellow
    
    if ($script:AllResults.Count -gt 0) {
        Export-MarkdownReport -Results $script:AllResults
        Show-FinalSummary -Results $script:AllResults
    } else {
        Write-Error "No test results collected!"
    }
}

# ============================================================================
# ENTRY POINT
# ============================================================================

try {
    Main
} catch {
    Write-Error "Fatal error during execution: $_"
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
} finally {
    Write-Host "`nScript execution finished at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
}
