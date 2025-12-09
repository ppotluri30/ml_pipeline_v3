# SLO-Driven Autoscaling Deployment Guide

## Overview

This guide covers deploying SLO-based autoscaling for the inference service using custom metrics (P95 latency, in-flight requests) instead of CPU-only scaling.

**SLO Targets:**
- P95 latency < 350ms
- In-flight requests < 10 per pod
- CPU < 80% (backstop)
- Error rate < 1%

## Prerequisites

### 1. Prometheus Setup

Ensure Prometheus is installed with the correct service monitors:

```bash
# Check if Prometheus is running
kubectl get pods -n monitoring | grep prometheus

# If not installed, install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

### 2. Choose Metrics Adapter (Option A or B)

#### Option A: Prometheus Adapter (Standard HPA)

```bash
# Install Prometheus Adapter
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring \
  --set prometheus.url=http://prometheus-kube-prometheus-prometheus.monitoring.svc \
  --set prometheus.port=9090
```

Create custom metrics configuration:

```yaml
# prometheus-adapter-values.yaml
rules:
- seriesQuery: 'http_request_duration_seconds_bucket{endpoint="/predict"}'
  resources:
    overrides:
      pod: {resource: "pod"}
  name:
    matches: "^(.*)_bucket$"
    as: "http_request_duration_p95"
  metricsQuery: |
    histogram_quantile(0.95,
      sum by (pod) (
        rate(<<.Series>>{<<.LabelMatchers>>}[1m])
      )
    )

- seriesQuery: 'http_requests_in_flight{endpoint="/predict"}'
  resources:
    overrides:
      pod: {resource: "pod"}
  name:
    as: "http_requests_in_flight"
  metricsQuery: |
    sum by (pod) (<<.Series>>{<<.LabelMatchers>>})
```

Update adapter:

```bash
helm upgrade prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring \
  -f prometheus-adapter-values.yaml
```

#### Option B: KEDA (Simpler Alternative)

```bash
# Install KEDA
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace

# Apply KEDA scaler
kubectl apply -f .kubernetes/inference-keda-scaler.yaml
```

**Skip to Step 4 if using KEDA** (no need for HPA manifest).

## Deployment Steps

### Step 1: Rebuild Inference Container with New Metrics

The `api_server.py` has been updated with HTTP-level metrics. Rebuild the image:

```bash
# From project root
docker-compose build inference

# Or for Kubernetes
docker build -t inference:slo-v1 ./inference_container
docker tag inference:slo-v1 your-registry/inference:slo-v1
docker push your-registry/inference:slo-v1
```

### Step 2: Deploy Updated Inference Service

```bash
# Apply new deployment with metrics port, resource requests, and probes
kubectl apply -f .kubernetes/inference-deployment-slo.yaml

# Verify pods are starting
kubectl get pods -l app=inference -w

# Check pod readiness (should wait for model to load)
kubectl describe pod -l app=inference | grep -A 5 Readiness

# Verify metrics endpoint is responding
kubectl port-forward svc/inference 8000:8000
curl http://localhost:8000/metrics | grep http_request_duration
```

Expected metrics output:
```
http_request_duration_seconds_bucket{method="POST",endpoint="/predict",status="200",le="0.35"} 42
http_requests_in_flight{method="POST",endpoint="/predict"} 3
http_requests_total{method="POST",endpoint="/predict",status="200"} 156
```

### Step 3: Apply Prometheus Recording Rules and Alerts

```bash
# Apply ServiceMonitor, PrometheusRule (bundled in HPA manifest)
kubectl apply -f .kubernetes/inference-slo-hpa.yaml

# Verify ServiceMonitor is created
kubectl get servicemonitor inference-metrics

# Verify PrometheusRule is created
kubectl get prometheusrule inference-slo-rules

# Check that Prometheus is scraping (after 15-30 seconds)
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
# Open http://localhost:9090 and query: http_request_duration_p95:pod
```

### Step 4: Deploy HPA (Option A) or Verify KEDA (Option B)

#### Option A: Standard HPA with Prometheus Adapter

```bash
# Apply HPA (if not already applied in Step 3)
kubectl apply -f .kubernetes/inference-slo-hpa.yaml

# Verify HPA is created
kubectl get hpa inference-slo-hpa

# Check HPA status
kubectl describe hpa inference-slo-hpa
```

Expected output:
```
Metrics:                                           
  - type: Pods
    pods:
      metric:
        name: http_request_duration_p95
      target:
        type: AverageValue
        averageValue: 350m
    current:
      averageValue: 120m
  
  - type: Pods
    pods:
      metric:
        name: http_requests_in_flight
      target:
        type: AverageValue
        averageValue: 10
    current:
      averageValue: 2
  
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80
    current:
      averageUtilization: 35%

Replicas: 3 current / 3 desired
```

#### Option B: KEDA Scaler

```bash
# Check KEDA scaler status
kubectl get scaledobject inference-keda-slo -n default

# Describe scaler
kubectl describe scaledobject inference-keda-slo

# Check HPA created by KEDA
kubectl get hpa keda-hpa-inference-keda-slo
```

### Step 5: Validate Metrics Flow

```bash
# Check that custom metrics are available
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/pods/*/http_request_duration_p95" | jq .

# Check in-flight metrics
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/pods/*/http_requests_in_flight" | jq .
```

Expected output:
```json
{
  "kind": "MetricValueList",
  "items": [
    {
      "describedObject": {"kind": "Pod", "name": "inference-abc123"},
      "metricName": "http_request_duration_p95",
      "value": "120m"  // 120ms
    }
  ]
}
```

### Step 6: Run Load Tests to Validate Autoscaling

Use the existing HPA test script:

```powershell
# From scripts directory
.\k8s_auto_hpa_tests.ps1 `
  -UserCounts 50,100,200 `
  -WorkerCounts 4 `
  -TestDurationSeconds 240 `
  -HPAMinReplicas 3 `
  -HPAMaxReplicas 20 `
  -HPATargetCPU 70 `
  -InitialReplicas 3 `
  -OutputDir "c:\Users\ppotluri\Downloads\ml_pipeline_v3\flts-main\results\slo_validation"
```

**Expected Behavior:**
1. **Baseline (50 users)**: 3 replicas, P95 latency ~150-200ms, no scaling
2. **Medium load (100 users)**: Scale to 5-6 replicas as P95 approaches 350ms
3. **High load (200 users)**: Scale to 10-12 replicas, P95 stays below 400ms
4. **Cool-down**: Slow scale-down over 5 minutes after load drops

### Step 7: Monitor Scaling Events

```bash
# Watch HPA scaling decisions in real-time
kubectl get hpa inference-slo-hpa -w

# View scaling events
kubectl get events --sort-by='.lastTimestamp' | grep -i scale

# Check pod count over time
watch -n 5 'kubectl get pods -l app=inference | wc -l'
```

### Step 8: Verify Alerting

Trigger alerts by generating load:

```bash
# High latency alert (P95 > 350ms for 2 minutes)
# Use Locust with 200 users, 8 workers for 3+ minutes

# Check Prometheus alerts
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
# Navigate to http://localhost:9090/alerts
```

Expected alerts:
- **InferenceLatencySLOBreach**: Fires when P95 > 350ms for 2 minutes
- **InferenceHighErrorRate**: Fires when error rate > 1% for 5 minutes

## Configuration Tuning

### Adjusting SLO Targets

Edit `.kubernetes/inference-slo-hpa.yaml`:

```yaml
metrics:
- type: Pods
  pods:
    metric:
      name: http_request_duration_p95
    target:
      averageValue: "500m"  # Change from 350ms to 500ms
```

### Tuning Scaling Behavior

**Faster scale-up** (for spiky traffic):

```yaml
scaleUp:
  stabilizationWindowSeconds: 30  # Reduced from 60
  policies:
  - type: Pods
    value: 4  # Add 4 pods at once instead of 2
    periodSeconds: 30
```

**Faster scale-down** (to save costs):

```yaml
scaleDown:
  stabilizationWindowSeconds: 180  # Reduced from 300
  policies:
  - type: Percent
    value: 20  # Remove 20% instead of 10%
    periodSeconds: 60
```

### Adjusting Pod Concurrency

Edit `inference-deployment-slo.yaml`:

```yaml
env:
- name: PREDICT_MAX_CONCURRENCY
  value: "32"  # Increase from 16 if CPU allows

- name: UVICORN_WORKERS
  value: "4"  # Increase from 2 if memory allows
```

**Note**: More workers = more memory, higher concurrency = more CPU

### Resource Sizing

Current configuration:
- **Requests**: 500m CPU, 2Gi memory (for HPA calculations)
- **Limits**: 2000m CPU, 4Gi memory (to prevent OOM)

For **higher throughput** (fewer, larger pods):
```yaml
resources:
  requests:
    cpu: "1000m"  # 1 CPU
    memory: "4Gi"
  limits:
    cpu: "4000m"  # 4 CPUs
    memory: "8Gi"
```

For **lower latency** (more, smaller pods):
```yaml
resources:
  requests:
    cpu: "250m"  # 0.25 CPU
    memory: "1Gi"
  limits:
    cpu: "1000m"  # 1 CPU
    memory: "2Gi"
```

## Troubleshooting

### HPA Shows "Unknown" Metrics

**Symptom:**
```
Metrics:                                           ( current / target )
  "http_request_duration_p95" on pods:  <unknown> / 350m
```

**Causes:**
1. Prometheus Adapter not installed or misconfigured
2. ServiceMonitor not scraping pods
3. Recording rules not evaluating

**Fix:**
```bash
# Check adapter logs
kubectl logs -n monitoring -l app.kubernetes.io/name=prometheus-adapter

# Verify custom metrics API
kubectl get apiservice v1beta1.custom.metrics.k8s.io

# Check if metrics appear in Prometheus
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
# Query: http_request_duration_p95:pod
```

### Pods Not Ready (Stuck in Init)

**Symptom:** Pods stuck in `Init:0/1` or `0/1 Running`

**Causes:**
1. Model download taking too long
2. Readiness probe failing before model loads
3. Memory/CPU insufficient for model

**Fix:**
```bash
# Check pod logs
kubectl logs -l app=inference --tail=100

# Check events
kubectl describe pod -l app=inference | grep -A 10 Events

# Increase startup probe failure threshold
# Edit inference-deployment-slo.yaml:
startupProbe:
  failureThreshold: 24  # 240 seconds instead of 120
```

### HPA Not Scaling Up Under Load

**Symptom:** P95 latency exceeds 350ms but replicas stay at minimum

**Causes:**
1. All metrics below threshold (need to check OR semantics)
2. HPA in cooldown/stabilization window
3. Max replicas already reached

**Fix:**
```bash
# Check current metric values
kubectl describe hpa inference-slo-hpa

# Check if maxReplicas is too low
kubectl get hpa inference-slo-hpa -o yaml | grep maxReplicas

# Reduce stabilization window temporarily
kubectl patch hpa inference-slo-hpa --type=merge -p '
spec:
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0
'
```

### High Error Rate Despite Scaling

**Symptom:** 5xx errors even after scaling up

**Causes:**
1. Model not loaded in new pods (readiness probe issue)
2. Backend dependencies slow (Kafka, MinIO, MLflow)
3. Input validation errors (400s counted as 5xx)

**Fix:**
```bash
# Check error breakdown
kubectl port-forward svc/inference 8000:8000
curl http://localhost:8000/metrics | grep http_request_errors

# Check which pods are serving errors
kubectl logs -l app=inference --tail=50 | grep "error_type"

# Ensure readiness probe is strict
# Edit deployment: READINESS_CHECK_MODEL=1
```

### KEDA vs Prometheus Adapter Decision

**Use KEDA if:**
- Simpler setup preferred
- Multiple scaling sources (Kafka lag + Prometheus)
- Event-driven workloads (scale to zero)

**Use Prometheus Adapter if:**
- Standard Kubernetes HPA preferred
- Already using Prometheus Operator
- Need fine-grained metric transformations

## Validation Checklist

Before declaring success:

- [ ] Metrics endpoint responds: `curl http://<pod-ip>:8000/metrics | grep http_request_duration`
- [ ] Prometheus scraping: Query `http_request_duration_p95:pod` returns data
- [ ] Custom metrics API: `kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1"` works
- [ ] HPA shows current values: `kubectl describe hpa` shows non-"unknown" metrics
- [ ] Load test triggers scale-up: 100+ users → replicas increase
- [ ] P95 latency stays below 500ms under load (350ms target + 150ms headroom)
- [ ] Slow scale-down after load: Takes 5+ minutes to return to min replicas
- [ ] Zero 5xx errors during scaling events
- [ ] Alerts fire correctly: Latency alert at 2min, error alert at 5min
- [ ] Pod anti-affinity working: Pods spread across nodes

## Recommended Defaults (Production)

Based on validation testing:

**HPA Configuration:**
- Min replicas: **3** (for availability during node failures)
- Max replicas: **20** (adjust based on cluster capacity)
- P95 latency target: **350ms** (user experience threshold)
- In-flight target: **10 per pod** (queue depth limit)
- CPU target: **80%** (backstop only)

**Pod Configuration:**
- Uvicorn workers: **2-4** (based on CPU cores)
- Max concurrency: **16-32** (based on model complexity)
- CPU request: **500m-1000m** (0.5-1 core)
- Memory request: **2-4Gi** (model + data)
- CPU limit: **2000m-4000m** (2-4 cores)
- Memory limit: **4-8Gi** (prevent OOM)

**Scaling Behavior:**
- Scale-up stabilization: **60 seconds** (avoid thrashing)
- Scale-up rate: **2 pods/30s OR 50%/30s**
- Scale-down stabilization: **300 seconds** (5 minutes)
- Scale-down rate: **1 pod/60s OR 10%/120s**

**Probe Timings:**
- Startup probe: **120 seconds max** (12 × 10s)
- Readiness probe: **10s period** (check model loaded)
- Liveness probe: **30s period** (check service responsive)

## Next Steps

1. **Create Grafana Dashboard** for real-time monitoring:
   - P95 latency vs target (350ms line)
   - Replica count over time
   - In-flight requests per pod
   - Error rate percentage
   - HPA scaling events timeline

2. **Set Up Alerting** (if not using Prometheus Operator):
   - PagerDuty/Slack integration for critical alerts
   - Latency SLO breach → Page on-call
   - High error rate → Page on-call
   - Scaling at max replicas → Warn capacity

3. **Load Test Matrix** with script:
   ```powershell
   .\k8s_auto_hpa_tests.ps1 `
     -UserCounts 25,50,100,200,400 `
     -WorkerCounts 4,8 `
     -TestDurationSeconds 240 `
     -HPAMinReplicas 3 `
     -HPAMaxReplicas 20
   ```

4. **Cost Analysis**:
   - Measure average replica count over 24 hours
   - Compare CPU-only scaling vs SLO-driven
   - Calculate cost per 1000 predictions

5. **Documentation Updates**:
   - Add SLO targets to main README
   - Document runbook for on-call (this guide)
   - Create capacity planning spreadsheet

## References

- Kubernetes HPA v2: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- Prometheus Adapter: https://github.com/kubernetes-sigs/prometheus-adapter
- KEDA: https://keda.sh/docs/
- Custom Metrics API: https://github.com/kubernetes/metrics
