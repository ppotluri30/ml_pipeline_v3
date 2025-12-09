# Real-Time Autoscaling Dashboard
# Monitors inference pod count, p95 latency, KEDA status, and scaling decisions
#
# Usage: .\Monitor-LiveLatency.ps1 -Duration 300 -RefreshInterval 2
#
# Displays live dashboard with:
# - Current pod count
# - p95 latency (Prometheus)
# - KEDA Active status
# - Target threshold
# - Scaling decision (Scale Up/Down/Stable)

param(
    [int]$Duration = 300,              # Total monitoring duration in seconds (default 5 min)
    [int]$RefreshInterval = 2,         # Dashboard refresh interval in seconds
    [string]$Namespace = "default",
    [string]$PrometheusDeployment = "prometheus-server",
    [string]$PrometheusContainer = "prometheus-server",
    [string]$PrometheusPort = "9090",
    [string]$ScaledObjectName = "inference-slo-scaler",
    [string]$DeploymentName = "inference",
    [double]$LatencyThreshold = 0.8,   # Threshold in seconds
    [switch]$Verbose
)

Clear-Host
Write-Host "Starting Real-Time Autoscaling Dashboard..." -ForegroundColor Cyan
Write-Host "Duration: $Duration seconds | Refresh: $RefreshInterval seconds" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
Write-Host ""
Start-Sleep -Seconds 2

# Helper function to get current pod count
function Get-PodCount {
    try {
        $replicas = kubectl get deployment $script:DeploymentName -n $script:Namespace -o jsonpath='{.status.replicas}' 2>$null
        if ($replicas) {
            return [int]$replicas
        } else {
            return "Error"
        }
    } catch {
        return "Error"
    }
}

# Helper function to query Prometheus p95 latency (returns seconds)
function Get-PrometheusP95Seconds {
    try {
        $query = "histogram_quantile(0.95,sum(rate(inference_latency_seconds_bucket[2m]))by(le))"
        $encodedQuery = [Uri]::EscapeDataString($query)
        
        $prometheusResponse = kubectl exec -n $script:Namespace deployment/$script:PrometheusDeployment -c $script:PrometheusContainer -- `
            wget -qO- "http://localhost:$($script:PrometheusPort)/api/v1/query?query=$encodedQuery" 2>$null
        
        if (-not $prometheusResponse) {
            return $null
        }
        
        $promData = $prometheusResponse | ConvertFrom-Json
        
        if ($promData.data.result.Count -gt 0) {
            $latencySeconds = [double]$promData.data.result[0].value[1]
            
            # Check for NaN or Infinity
            if ([double]::IsNaN($latencySeconds) -or [double]::IsInfinity($latencySeconds)) {
                return $null
            }
            
            return $latencySeconds
        } else {
            return $null
        }
    } catch {
        return $null
    }
}

# Helper function to get KEDA Active status
function Get-KEDAStatus {
    try {
        $scaledObjectJson = kubectl get scaledobject $script:ScaledObjectName -n $script:Namespace -o json 2>$null | ConvertFrom-Json
        
        if ($scaledObjectJson) {
            $activeCondition = $scaledObjectJson.status.conditions | Where-Object { $_.type -eq 'Active' }
            if ($activeCondition) {
                return $activeCondition.status
            } else {
                return "Unknown"
            }
        } else {
            return "NotFound"
        }
    } catch {
        return "Error"
    }
}

# Helper function to get Prometheus last value from KEDA external metric
function Get-KEDAPrometheusValue {
    try {
        $labelSelector = [Uri]::EscapeDataString("scaledobject.keda.sh/name=$($script:ScaledObjectName)")
        $kedaMetric = kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/$($script:Namespace)/s0-prometheus?labelSelector=$labelSelector" 2>$null | ConvertFrom-Json
        
        if ($kedaMetric.items.Count -gt 0) {
            $valueString = $kedaMetric.items[0].value
            # Parse millicores format (e.g., "1864m" = 1.864 seconds)
            if ($valueString -match "^(-?\d+)m$") {
                $milliseconds = [int]$Matches[1]
                $seconds = $milliseconds / 1000.0
                return $seconds
            } else {
                return $null
            }
        } else {
            return $null
        }
    } catch {
        return $null
    }
}

# Set script-level variables for helper functions
$script:Namespace = $Namespace
$script:PrometheusDeployment = $PrometheusDeployment
$script:PrometheusContainer = $PrometheusContainer
$script:PrometheusPort = $PrometheusPort
$script:ScaledObjectName = $ScaledObjectName
$script:DeploymentName = $DeploymentName
$script:LatencyThreshold = $LatencyThreshold
$script:Verbose = $Verbose

# Main monitoring loop
$startTime = Get-Date
$endTime = $startTime.AddSeconds($Duration)

try {
    while ((Get-Date) -lt $endTime) {
        # Clear screen for dashboard effect
        Clear-Host
        
        # Get current time
        $currentTime = Get-Date -Format "HH:mm:ss"
        
        # Collect metrics
        $podCount = Get-PodCount
        $latencySeconds = Get-PrometheusP95Seconds
        $kedaStatus = Get-KEDAStatus
        $kedaPrometheusValue = Get-KEDAPrometheusValue
        
        # Format latency display
        $latencyDisplay = if ($null -ne $latencySeconds) {
            [math]::Round($latencySeconds, 2)
        } else {
            "N/A (need traffic)"
        }
        
        # Format KEDA status
        $kedaStatusDisplay = if ($kedaStatus -eq "True") {
            "Active"
        } elseif ($kedaStatus -eq "False") {
            "Inactive"
        } else {
            $kedaStatus
        }
        
        # Format KEDA Prometheus value
        $kedaPrometheusDisplay = if ($null -ne $kedaPrometheusValue) {
            "$([math]::Round($kedaPrometheusValue, 2))s"
        } else {
            "N/A"
        }
        
        # Determine scaling decision
        $scalingDecision = "Stable"
        $decisionColor = "Green"
        
        if ($null -ne $latencySeconds) {
            if ($latencySeconds -gt $script:LatencyThreshold) {
                if ($kedaStatus -eq "True") {
                    $scalingDecision = "Scale Up (Active)"
                    $decisionColor = "Yellow"
                } else {
                    $scalingDecision = "Scale Up (Pending)"
                    $decisionColor = "Yellow"
                }
            } elseif ($latencySeconds -lt ($script:LatencyThreshold * 0.5)) {
                $scalingDecision = "Scale Down (Cooldown)"
                $decisionColor = "Cyan"
            }
        }
        
        # Display dashboard
        Write-Host "──────────────── AUTOSCALING REAL-TIME MONITOR ────────────────" -ForegroundColor Cyan
        Write-Host ("Time: {0,-50}" -f $currentTime) -ForegroundColor White
        Write-Host ""
        Write-Host ("Pod Replicas:                   {0,-30}" -f $podCount) -ForegroundColor White
        Write-Host ("Latency P95 (seconds):          {0,-30}" -f $latencyDisplay) -ForegroundColor $(if ($null -ne $latencySeconds -and $latencySeconds -gt $script:LatencyThreshold) { "Red" } else { "White" })
        Write-Host ("KEDA Status:                    {0,-30}" -f $kedaStatusDisplay) -ForegroundColor $(if ($kedaStatus -eq "True") { "Green" } else { "Gray" })
        Write-Host ("Target Threshold:               > {0}s" -f $script:LatencyThreshold) -ForegroundColor Gray
        Write-Host ("Last Prometheus Value:          {0,-30}" -f $kedaPrometheusDisplay) -ForegroundColor Gray
        Write-Host ""
        Write-Host ("Scaling Decision:               {0,-30}" -f $scalingDecision) -ForegroundColor $decisionColor
        Write-Host "────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Refresh: ${RefreshInterval}s | Duration: $([math]::Round(($endTime - (Get-Date)).TotalSeconds, 0))s remaining | Ctrl+C to stop" -ForegroundColor Gray
        
        # Wait before next refresh
        Start-Sleep -Seconds $RefreshInterval
    }
} catch {
    Write-Host "`nMonitoring stopped." -ForegroundColor Yellow
}

Write-Host "`nMonitoring complete!" -ForegroundColor Green
