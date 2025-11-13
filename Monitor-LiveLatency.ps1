# Real-Time Latency Monitoring Dashboard
# Monitors Prometheus, KEDA, HPA, Locust, and Pod metrics during load tests
#
# Usage: .\Monitor-LiveLatency.ps1 -Duration 120 -SampleInterval 5
#
# Displays live table with:
# - Timestamp
# - Locust p95 (from /stats/requests)
# - Prometheus p95 (from histogram_quantile query)
# - KEDA external metric (s0-prometheus)
# - Current replicas
# - Desired replicas
# - CPU utilization
# - Queue length
# - Scaling decision/status

param(
    [int]$Duration = 120,        # Total monitoring duration in seconds
    [int]$SampleInterval = 5,    # Sampling interval in seconds
    [string]$OutputCsv = "latency-monitoring-$(Get-Date -Format 'yyyyMMdd-HHmmss').csv"
)

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host " Real-Time Latency Monitoring Dashboard" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "Duration: $Duration seconds" -ForegroundColor Yellow
Write-Host "Sample Interval: $SampleInterval seconds" -ForegroundColor Yellow
Write-Host "Output File: $OutputCsv" -ForegroundColor Yellow
Write-Host ""

# Initialize CSV header
$csvHeader = "Timestamp,LocustP95ms,PrometheusP95ms,KEDALatencyms,CurrentReplicas,DesiredReplicas,CPUPercent,QueueLength,ScalingStatus"
$csvHeader | Out-File -FilePath $OutputCsv -Encoding utf8

# Helper function to safely parse Locust stats
function Get-LocustStats {
    try {
        # Try to reach Locust via kubectl port-forward (assume it's running on 8089)
        $locustStats = Invoke-RestMethod -Uri "http://localhost:8089/stats/requests" -Method GET -TimeoutSec 3 -ErrorAction Stop
        
        # Find the /predict endpoint stats or aggregated stats
        $predictStats = $locustStats.stats | Where-Object { 
            $_.name -eq "/predict" -or 
            $_.name -eq "POST /predict" -or 
            $_.name -eq "Aggregated" 
        } | Select-Object -First 1
        
        if ($predictStats) {
            # Try multiple field name variations for p95
            $p95Value = $null
            
            # Try: response_time_percentile_95 (no decimal)
            if ($predictStats.PSObject.Properties.Name -contains "response_time_percentile_95") {
                $p95Value = $predictStats.response_time_percentile_95
            }
            # Try: 95%ile (Locust 2.x format)
            elseif ($predictStats.PSObject.Properties.Name -contains "95%ile") {
                $p95Value = $predictStats.'95%ile'
            }
            # Try: response_time_percentile_0.95 (with decimal)
            else {
                $p95Field = $predictStats.PSObject.Properties | Where-Object { $_.Name -match "response.*95" }
                if ($p95Field) {
                    $p95Value = $p95Field.Value
                }
            }
            
            if ($null -ne $p95Value -and $p95Value -ne 0) {
                # Locust returns milliseconds already
                return [math]::Round([double]$p95Value, 0)
            } else {
                return "N/A"
            }
        } else {
            return "N/A"
        }
    } catch {
        # Port-forward might not be running, return N/A silently
        return "N/A"
    }
}

# Helper function to ensure port-forward is running
function Ensure-LocustPortForward {
    param([ref]$PortForwardJob)
    
    try {
        # Test if port-forward is already working
        $null = Invoke-RestMethod -Uri "http://localhost:8089/stats/requests" -Method GET -TimeoutSec 2 -ErrorAction Stop
        return $true
    } catch {
        # Port-forward not working, start it
        if ($PortForwardJob.Value) {
            # Kill existing job if any
            Stop-Job -Job $PortForwardJob.Value -ErrorAction SilentlyContinue
            Remove-Job -Job $PortForwardJob.Value -Force -ErrorAction SilentlyContinue
        }
        
        # Start new port-forward job
        $PortForwardJob.Value = Start-Job -ScriptBlock {
            kubectl port-forward svc/locust-master 8089:8089 2>&1 | Out-Null
        }
        
        # Wait for port-forward to establish
        Start-Sleep -Seconds 3
        return $false
    }
}

# Helper function to query Prometheus p95 latency
function Get-PrometheusP95 {
    try {
        $query = "histogram_quantile(0.95,sum(rate(inference_latency_seconds_bucket[2m]))by(le))"
        $encodedQuery = [System.Web.HttpUtility]::UrlEncode($query)
        
        $prometheusResponse = kubectl exec deployment/prometheus-server -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$encodedQuery" 2>$null
        
        $promData = $prometheusResponse | ConvertFrom-Json
        
        if ($promData.data.result.Count -gt 0) {
            $latencySeconds = [double]$promData.data.result[0].value[1]
            
            # Check for NaN or Infinity
            if ([double]::IsNaN($latencySeconds) -or [double]::IsInfinity($latencySeconds)) {
                return "NaN"
            }
            
            $latencyMs = [math]::Round($latencySeconds * 1000, 0)
            return $latencyMs
        } else {
            return "Empty"
        }
    } catch {
        return "Error"
    }
}

# Helper function to get KEDA external metric
function Get-KEDALatency {
    try {
        $kedaMetric = kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/default/s0-prometheus?labelSelector=scaledobject.keda.sh%2Fname%3Dinference-slo-scaler" 2>$null | ConvertFrom-Json
        
        if ($kedaMetric.items.Count -gt 0) {
            $valueString = $kedaMetric.items[0].value
            # Parse millicores format (e.g., "1864m" = 1.864 seconds = 1864ms)
            if ($valueString -match "^(-?\d+)m$") {
                $ms = [int]$Matches[1]
                return $ms
            } else {
                return "InvalidFormat"
            }
        } else {
            return "NoData"
        }
    } catch {
        return "Error"
    }
}

# Helper function to get HPA metrics
function Get-HPAMetrics {
    try {
        $hpaJson = kubectl get hpa keda-hpa-inference-slo-scaler -o json 2>$null | ConvertFrom-Json
        
        $currentReplicas = $hpaJson.status.currentReplicas
        $desiredReplicas = $hpaJson.status.desiredReplicas
        
        # Extract CPU percentage
        $cpuMetric = $hpaJson.status.currentMetrics | Where-Object { $_.type -eq 'Resource' -and $_.resource.name -eq 'cpu' }
        $cpuPercent = if ($cpuMetric) { $cpuMetric.resource.current.averageUtilization } else { "N/A" }
        
        # Extract queue length
        $queueMetric = $hpaJson.status.currentMetrics | Where-Object { $_.type -eq 'External' -and $_.external.metric.name -eq 's1-prometheus' }
        $queueLength = if ($queueMetric) {
            $queueValue = $queueMetric.external.current.averageValue
            if ($queueValue -match "^(\d+)m?$") {
                [int]$Matches[1]
            } else {
                "N/A"
            }
        } else {
            "N/A"
        }
        
        # Determine scaling status
        $scalingStatus = "Stable"
        if ($currentReplicas -lt $desiredReplicas) {
            $scalingStatus = "ScalingUp"
        } elseif ($currentReplicas -gt $desiredReplicas) {
            $scalingStatus = "ScalingDown"
        }
        
        $conditions = $hpaJson.status.conditions | Where-Object { $_.type -eq 'ScalingLimited' }
        if ($conditions -and $conditions.status -eq 'True') {
            $scalingStatus += "-Limited"
        }
        
        return @{
            CurrentReplicas = $currentReplicas
            DesiredReplicas = $desiredReplicas
            CPUPercent = $cpuPercent
            QueueLength = $queueLength
            ScalingStatus = $scalingStatus
        }
    } catch {
        return @{
            CurrentReplicas = "Error"
            DesiredReplicas = "Error"
            CPUPercent = "Error"
            QueueLength = "Error"
            ScalingStatus = "Error"
        }
    }
}

# Display table header
Write-Host ""
Write-Host ("=" * 160) -ForegroundColor Cyan
Write-Host ("{0,-20} {1,12} {2,14} {3,14} {4,10} {5,10} {6,10} {7,12} {8,-20}" -f `
    "Timestamp", "LocustP95", "PromP95", "KEDALatency", "Current", "Desired", "CPU%", "QueueLen", "Status") -ForegroundColor White
Write-Host ("=" * 160) -ForegroundColor Cyan

# Start port-forward for Locust (optional, but helpful)
$portForwardJob = $null
Write-Host "Starting Locust port-forward..." -ForegroundColor Yellow
Ensure-LocustPortForward -PortForwardJob ([ref]$portForwardJob)
Write-Host ""

# Main monitoring loop
$startTime = Get-Date
$endTime = $startTime.AddSeconds($Duration)
$iteration = 0

while ((Get-Date) -lt $endTime) {
    $iteration++
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    # Collect all metrics
    $locustP95 = Get-LocustStats
    $promP95 = Get-PrometheusP95
    
    # Retry Prometheus if NaN on first few attempts (warmup period)
    if ($promP95 -eq "NaN" -and $iteration -le 3) {
        Write-Host "⚠️ Prometheus warmup period - waiting for sufficient samples..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
        $promP95 = Get-PrometheusP95
    }
    
    $kedaLatency = Get-KEDALatency
    $hpaMetrics = Get-HPAMetrics
    
    # Format values for display
    $locustDisplay = if ($locustP95 -eq "N/A") { "-" } else { "$locustP95 ms" }
    $promDisplay = if ($promP95 -in @("Empty", "Error", "NaN")) { $promP95 } else { "$promP95 ms" }
    $kedaDisplay = if ($kedaLatency -in @("NoData", "Error", "InvalidFormat")) { $kedaLatency } else { "$kedaLatency ms" }
    
    # Determine row color based on status
    $rowColor = "Gray"
    if ($hpaMetrics.ScalingStatus -match "ScalingUp") {
        $rowColor = "Green"
    } elseif ($hpaMetrics.ScalingStatus -match "ScalingDown") {
        $rowColor = "Yellow"
    } elseif ($promP95 -ne "Empty" -and $promP95 -ne "Error" -and $promP95 -ne "NaN" -and $promP95 -gt 2000) {
        $rowColor = "Red"  # Latency above 2s threshold
    }
    
    # Display row
    Write-Host ("{0,-20} {1,12} {2,14} {3,14} {4,10} {5,10} {6,10} {7,12} {8,-20}" -f `
        $timestamp, `
        $locustDisplay, `
        $promDisplay, `
        $kedaDisplay, `
        $hpaMetrics.CurrentReplicas, `
        $hpaMetrics.DesiredReplicas, `
        $hpaMetrics.CPUPercent, `
        $hpaMetrics.QueueLength, `
        $hpaMetrics.ScalingStatus) -ForegroundColor $rowColor
    
    # Write to CSV
    $csvLine = "$timestamp,$locustP95,$promP95,$kedaLatency,$($hpaMetrics.CurrentReplicas),$($hpaMetrics.DesiredReplicas),$($hpaMetrics.CPUPercent),$($hpaMetrics.QueueLength),$($hpaMetrics.ScalingStatus)"
    $csvLine | Out-File -FilePath $OutputCsv -Append -Encoding utf8
    
    # Wait for next sample
    if ((Get-Date) -lt $endTime) {
        Start-Sleep -Seconds $SampleInterval
    }
}

Write-Host ("=" * 160) -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitoring complete! Data saved to: $OutputCsv" -ForegroundColor Green

# Clean up port-forward job
if ($portForwardJob) {
    Write-Host "Stopping Locust port-forward..." -ForegroundColor Yellow
    Stop-Job -Job $portForwardJob -ErrorAction SilentlyContinue
    Remove-Job -Job $portForwardJob -Force -ErrorAction SilentlyContinue
}

Write-Host ""

# Display summary statistics
Write-Host "Summary Statistics:" -ForegroundColor Cyan
$csvData = Import-Csv -Path $OutputCsv

$validPromP95 = $csvData | Where-Object { $_.PrometheusP95ms -match '^\d+$' } | Select-Object -ExpandProperty PrometheusP95ms | ForEach-Object { [int]$_ }
$validKedaLatency = $csvData | Where-Object { $_.KEDALatencyms -match '^-?\d+$' } | Select-Object -ExpandProperty KEDALatencyms | ForEach-Object { [int]$_ }
$validLocustP95 = $csvData | Where-Object { $_.LocustP95ms -match '^\d+$' } | Select-Object -ExpandProperty LocustP95ms | ForEach-Object { [int]$_ }

if ($validPromP95.Count -gt 0) {
    Write-Host "  Prometheus p95:  Min=$([math]::Round(($validPromP95 | Measure-Object -Minimum).Minimum, 0))ms, Max=$([math]::Round(($validPromP95 | Measure-Object -Maximum).Maximum, 0))ms, Avg=$([math]::Round(($validPromP95 | Measure-Object -Average).Average, 0))ms" -ForegroundColor White
}

if ($validKedaLatency.Count -gt 0) {
    Write-Host "  KEDA Latency:    Min=$([math]::Round(($validKedaLatency | Measure-Object -Minimum).Minimum, 0))ms, Max=$([math]::Round(($validKedaLatency | Measure-Object -Maximum).Maximum, 0))ms, Avg=$([math]::Round(($validKedaLatency | Measure-Object -Average).Average, 0))ms" -ForegroundColor White
}

if ($validLocustP95.Count -gt 0) {
    Write-Host "  Locust p95:      Min=$([math]::Round(($validLocustP95 | Measure-Object -Minimum).Minimum, 0))ms, Max=$([math]::Round(($validLocustP95 | Measure-Object -Maximum).Maximum, 0))ms, Avg=$([math]::Round(($validLocustP95 | Measure-Object -Average).Average, 0))ms" -ForegroundColor White
}

$replicaChanges = $csvData | Where-Object { $_.CurrentReplicas -ne $_.DesiredReplicas }
if ($replicaChanges.Count -gt 0) {
    Write-Host "  Scaling Events:  $($replicaChanges.Count) observed" -ForegroundColor Yellow
} else {
    Write-Host "  Scaling Events:  0 (stable)" -ForegroundColor Green
}

Write-Host ""
