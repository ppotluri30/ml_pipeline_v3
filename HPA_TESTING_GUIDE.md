# Kubernetes HPA Testing Guide

## Overview

The `k8s_auto_hpa_tests.ps1` script has been updated to use **HPA-driven dynamic scaling** instead of manual replica management. The script automatically configures and monitors the Horizontal Pod Autoscaler during load tests.

## Key Changes

### ✅ What Changed

1. **Removed manual replica scaling** - No more `kubectl scale deployment inference --replicas=N`
2. **Automatic HPA configuration** - Creates or updates HPA with desired min/max/target settings
3. **Dynamic scaling observation** - HPA scales replicas automatically based on CPU utilization
4. **Enhanced monitoring** - Tracks initial → peak → final replica counts and scaling events
5. **Simplified test matrix** - Removed `ReplicaCounts` parameter; tests focus on load variations

### 📊 New HPA Metrics Tracked

- **Initial Replicas**: Starting replica count before load test
- **Peak Replicas**: Maximum replicas reached during test
- **Final Replicas**: Replica count after test completion
- **Scale-Up Events**: Number of times HPA increased replicas
- **Scale-Down Events**: Number of times HPA decreased replicas
- **Average CPU**: Mean CPU utilization across test duration
- **Min/Max HPA Bounds**: HPA configuration limits

## Usage

### Basic Example

```powershell
# Run with default settings (min=2, max=20, targetCPU=70%)
.\scripts\k8s_auto_hpa_tests.ps1 -UserCounts @(50,100,200) -WorkerCounts @(4,8)
```

### Custom HPA Configuration

```powershell
# Configure HPA bounds and target
.\scripts\k8s_auto_hpa_tests.ps1 `
    -UserCounts @(100,200,400) `
    -WorkerCounts @(4,8) `
    -HPAMinReplicas 2 `
    -HPAMaxReplicas 15 `
    -HPATargetCPU 70 `
    -InitialReplicas 3 `
    -TestDuration 120
```

### Quick Validation Test

```powershell
# Single test to verify HPA behavior
.\scripts\k8s_auto_hpa_tests.ps1 `
    -UserCounts @(100) `
    -WorkerCounts @(4) `
    -TestDuration 60
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TestDuration` | 120 | Test duration in seconds |
| `InitialReplicas` | 2 | Baseline replicas before HPA takes over |
| `HPAMinReplicas` | 2 | Minimum replicas for HPA |
| `HPAMaxReplicas` | 20 | Maximum replicas for HPA |
| `HPATargetCPU` | 70 | Target CPU utilization percentage |
| `WorkerCounts` | @(4, 8) | Array of Locust worker counts |
| `UserCounts` | @(50, 100, 200, 400) | Array of concurrent user counts |
| `MonitoringInterval` | 10 | HPA metrics sampling interval (seconds) |

## How It Works

### 1. Pre-Test Setup

```
├─ Check Kubernetes connection
├─ Ensure HPA exists with correct configuration
│  └─ If missing: Create HPA
│  └─ If exists: Update min/max/targetCPU
├─ Set initial replica baseline
└─ Wait for pods to stabilize
```

### 2. Test Execution

```
For each worker count:
  For each user count:
    ├─ Record initial HPA state (replicas, CPU)
    ├─ Start background monitoring job
    │  └─ Sample HPA metrics every 10s
    │     ├─ Current replicas
    │     ├─ Desired replicas
    │     └─ CPU utilization
    ├─ Run Locust load test
    ├─ Wait for monitoring to complete
    └─ Analyze scaling events
```

### 3. Metrics Collection

The monitoring job samples HPA status during the test:

```powershell
kubectl get hpa inference -o json
# Extracts:
# - status.currentReplicas
# - status.desiredReplicas
# - status.currentMetrics[cpu].current.averageUtilization
```

### 4. Results Analysis

- **Scaling Events**: Detected by comparing replica counts across samples
- **Peak Replicas**: Maximum replica count reached
- **Average CPU**: Mean of all CPU samples
- **Scale Direction**: Up vs. down events

## Example Output

### Console Output

```
[HPA METRICS]
  Min/Max Replicas:  2/10
  Initial Replicas:  2
  Peak Replicas:     6
  Final Replicas:    4
  Avg CPU:           67.5%
  Scaling Events:    3 (2 scale-up, 1 scale-down)
```

### CSV Output

```csv
Workers,Users,InitialReplicas,PeakReplicas,FinalReplicas,MinReplicas,MaxReplicas,AvgCPU%,ScalingEvents,ScaleUpEvents,ScaleDownEvents,RPS,Median_ms,P95_ms,P99_ms,AvgLatency_ms,Failures,Failures_Pct,TotalRequests,Duration_s
4,50,2,3,3,2,3,61.1,1,1,0,16.36,1200,2700,3600,1292,0,0,966,60
```

### Markdown Report

```markdown
## Complete Results

| Workers | Users | Init→Peak→Final | Min/Max | Avg CPU% | Scaling Events | RPS | P50(ms) | P95(ms) |
|---------|-------|-----------------|---------|----------|----------------|-----|---------|---------|
| 4 | 50 | 2->3->3 | 2/3 | 61.1 | 1 (1up 0down) | 16.36 | 1200 | 2700 |
```

## HPA Behavior Expectations

### Typical Scaling Timeline

1. **0-15s**: Initial ramp-up, CPU increases
2. **15-30s**: CPU crosses threshold (70%), HPA triggers scale-up
3. **30-45s**: New pods starting, CPU still high
4. **45-60s**: New pods ready, load distributed, CPU stabilizes
5. **60-90s**: Test ends, CPU drops
6. **90-120s**: HPA scale-down delay, then gradual decrease

### Scale-Up Triggers

- **Condition**: CPU > 70% for ~15 seconds
- **Action**: HPA increases `desiredReplicas`
- **Result**: New pods scheduled and started
- **Timing**: 30-60 seconds for pods to become ready

### Scale-Down Triggers

- **Condition**: CPU < 70% for ~5 minutes (default stabilization window)
- **Action**: HPA decreases `desiredReplicas`
- **Result**: Pods terminated gracefully
- **Timing**: Gradual, respects `scaleDownStabilizationWindowSeconds`

## Validation Checklist

After running tests, verify:

- ✅ HPA configuration matches parameters (min/max/target)
- ✅ Initial replicas set correctly before test
- ✅ Scaling events recorded when CPU > 70%
- ✅ Peak replicas reached under high load
- ✅ Final replicas stabilize after test
- ✅ CSV/Markdown reports include HPA metrics
- ✅ No test failures (0% failure rate)

## Troubleshooting

### HPA Not Scaling

**Problem**: Replicas stay at initial count despite high load

**Solutions**:
```powershell
# Check HPA status
kubectl get hpa inference
kubectl describe hpa inference

# Verify metrics-server is running
kubectl get pods -n kube-system | Select-String metrics-server

# Check pod CPU requests are set
kubectl get deployment inference -o yaml | Select-String -Pattern "resources:" -Context 0,5
```

### Monitoring Job Fails

**Problem**: `Get-MetricsSummary` returns zeros

**Solutions**:
```powershell
# Test HPA access manually
kubectl get hpa inference -o json | ConvertFrom-Json

# Check if deployment has readyReplicas
kubectl get deployment inference -o jsonpath='{.status.readyReplicas}'
```

### Slow Scaling

**Problem**: HPA takes > 2 minutes to scale up

**Solutions**:
```powershell
# Reduce HPA sync period (default 15s)
kubectl patch hpa inference --type merge -p '{"spec":{"behavior":{"scaleUp":{"stabilizationWindowSeconds":0}}}}'

# Increase target CPU to trigger faster
.\scripts\k8s_auto_hpa_tests.ps1 -HPATargetCPU 60
```

## Best Practices

1. **Set resource requests**: Ensure deployment has CPU requests defined
   ```yaml
   resources:
     requests:
       cpu: "100m"
   ```

2. **Allow stabilization time**: Use 30s cooldown between tests
   ```powershell
   -TestDuration 120  # Longer tests capture full scaling behavior
   ```

3. **Monitor metrics-server**: Verify it's collecting pod metrics
   ```bash
   kubectl top pods -l app=inference
   ```

4. **Test incrementally**: Start with low load to verify baseline
   ```powershell
   -UserCounts @(50,100,200,400)  # Gradual load increase
   ```

5. **Review logs**: Check HPA controller logs if behavior is unexpected
   ```bash
   kubectl logs -n kube-system -l k8s-app=kube-controller-manager | grep HorizontalPodAutoscaler
   ```

## Comparison: Before vs. After

### Before (Manual Scaling)

```powershell
foreach ($replicas in @(2,4,8)) {
    kubectl scale deployment inference --replicas=$replicas
    # Run test at fixed replica count
}
```

- ❌ No dynamic response to load
- ❌ Manual replica management
- ❌ Fixed capacity throughout test
- ❌ Doesn't reflect production behavior

### After (HPA-Driven)

```powershell
# HPA configured once
Ensure-HPA -MinReplicas 2 -MaxReplicas 20 -TargetCPU 70

# Tests observe natural scaling
foreach ($users in @(50,100,200,400)) {
    # HPA scales replicas based on actual CPU
}
```

- ✅ Dynamic scaling based on CPU
- ✅ Automatic capacity adjustment
- ✅ Realistic production simulation
- ✅ Observes scaling latency and behavior

## Results Interpretation

### Healthy HPA Behavior

```
Users: 100 → Init:2, Peak:4, Final:3
- Scale-up when CPU > 70% ✅
- Peak during load ✅
- Gradual scale-down after test ✅
- Average CPU ~65-75% ✅
```

### Potential Issues

```
Users: 200 → Init:2, Peak:2, Final:2, CPU:95%
- No scaling despite high CPU ❌
- Check: HPA exists? Metrics available? Max replicas too low?

Users: 400 → Init:2, Peak:20, Final:20, CPU:85%
- Hit max replicas, still overloaded ❌
- Action: Increase HPAMaxReplicas or reduce load

Users: 50 → Init:2, Peak:8, Final:2, CPU:30%
- Excessive scaling for low load ❌
- Action: Increase HPATargetCPU (e.g., 70% → 80%)
```

## See Also

- Script: `scripts/k8s_auto_hpa_tests.ps1`
- Results: `reports/k8s_hpa_performance/`
- Kubernetes HPA docs: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- FLTS Architecture: `.github/copilot-instructions.md`
