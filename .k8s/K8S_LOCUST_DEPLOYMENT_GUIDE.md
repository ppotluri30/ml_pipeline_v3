# Kubernetes Locust Load Testing - Deployment Guide

This guide covers deploying and running distributed Locust load tests natively in Kubernetes, replicating the functionality of `run_all_locust_tests.ps1`.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                          │
│                                                                 │
│  ┌──────────────┐     ┌──────────────────────────────────┐    │
│  │ Locust       │────▶│ Inference Service                 │    │
│  │ Master       │     │ (ClusterIP: 10.107.123.158:8000)  │    │
│  │ (NodePort    │     │                                    │    │
│  │  30089)      │     │  ┌─────────┐ ┌─────────┐          │    │
│  └──────────────┘     │  │ Pod 1   │ │ Pod 2   │ ...      │    │
│         │             │  └─────────┘ └─────────┘          │    │
│         │             └──────────────────────────────────────┘    │
│         │                                                        │
│  ┌──────▼─────────┐                                             │
│  │ Locust Workers │                                             │
│  │ (4-8 replicas) │                                             │
│  └────────────────┘                                             │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Driver Job (Orchestration)                             │    │
│  │ - Scales inference deployment (1,2,4,8 replicas)       │    │
│  │ - Scales worker deployment (4,8 workers)               │    │
│  │ - Starts tests via Locust API (200,400,800 users)     │    │
│  │ - Collects stats and generates reports                 │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Files Created

1. **locust-configmap.yaml** - ConfigMap with simplified Locust test script
2. **locust-master.yaml** - Locust master deployment + NodePort service (port 30089)
3. **locust-worker.yaml** - Scalable worker deployment (4-8 replicas)
4. **locust-driver.py** - Python orchestration script (test matrix automation)
5. **locust-driver-job.yaml** - Kubernetes Job with RBAC for driver execution
6. **Dockerfile.driver** - Custom image for driver (optional, recommended)

## Prerequisites

- Kubernetes cluster (Docker Desktop, Minikube, or production cluster)
- `kubectl` configured with cluster access
- Inference deployment running (tested with 2 replicas)
- At least 6GB RAM available for cluster

## Quick Start

### Option 1: Using Pre-built Locust Image (Fastest)

```bash
# 1. Deploy ConfigMap and Locust components
kubectl apply -f .k8s/locust-configmap.yaml
kubectl apply -f .k8s/locust-master.yaml
kubectl apply -f .k8s/locust-worker.yaml

# 2. Wait for Locust master and workers to be ready
kubectl wait --for=condition=ready pod -l app=locust,role=master --timeout=120s
kubectl wait --for=condition=ready pod -l app=locust,role=worker --timeout=120s

# 3. Verify Locust is accessible
curl http://localhost:30089/
# Should return Locust web UI HTML

# 4. Build and deploy driver (custom image recommended)
cd .k8s
docker build -t locust-driver:latest -f Dockerfile.driver .

# 5. Update locust-driver-job.yaml to use custom image
# Replace the complex Job definition with:
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: locust-load-test-matrix
  namespace: default
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      serviceAccountName: locust-driver
      restartPolicy: OnFailure
      containers:
      - name: driver
        image: locust-driver:latest
        imagePullPolicy: Never  # Use local image (Docker Desktop)
        volumeMounts:
        - name: results
          mountPath: /results
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
      - name: results
        emptyDir: {}
EOF

# 6. Apply RBAC for driver
kubectl apply -f .k8s/locust-driver-job.yaml  # Only RBAC parts

# 7. Start the test matrix
kubectl delete job locust-load-test-matrix --ignore-not-found
kubectl create job locust-load-test-matrix --image=locust-driver:latest \
  --dry-run=client -o yaml | kubectl apply -f -

# 8. Monitor progress
kubectl logs -f job/locust-load-test-matrix

# 9. Retrieve results when complete
POD=$(kubectl get pod -l job-name=locust-load-test-matrix -o jsonpath='{.items[0].metadata.name}')
kubectl cp $POD:/results/auto_summary.csv ./auto_summary.csv
kubectl cp $POD:/results/auto_summary.md ./auto_summary.md
```

### Option 2: Run Driver Locally (Development/Testing)

```bash
# 1. Deploy Locust components
kubectl apply -f .k8s/locust-configmap.yaml
kubectl apply -f .k8s/locust-master.yaml
kubectl apply -f .k8s/locust-worker.yaml

# 2. Wait for ready state
kubectl wait --for=condition=ready pod -l app=locust --timeout=120s

# 3. Port-forward Locust master (if not using NodePort)
kubectl port-forward svc/locust-master 8089:8089 &

# 4. Install Python dependencies locally
pip install requests

# 5. Run driver script locally
cd .k8s
python locust-driver.py

# Results will be saved to ./results/ directory
```

### Option 3: Manual Testing (Single Scenario)

```bash
# 1. Scale infrastructure manually
kubectl scale deployment inference --replicas=2
kubectl scale deployment locust-worker --replicas=4

# 2. Wait for ready state
kubectl wait --for=condition=ready pod -l app=inference --timeout=120s
kubectl wait --for=condition=ready pod -l app=locust,role=worker --timeout=120s

# 3. Start test via Locust API
curl -X POST http://localhost:30089/swarm \
  -d "user_count=200&spawn_rate=20&host=http://inference:8000"

# 4. Monitor stats (poll every 2 seconds)
while true; do
  curl -s http://localhost:30089/stats/requests | jq '.stats[] | select(.name=="Aggregated")'
  sleep 2
done

# 5. Stop test after desired duration
curl http://localhost:30089/stop

# 6. Get final stats
curl http://localhost:30089/stats/requests | jq '.stats[] | select(.name=="Aggregated")'

# 7. Reset stats for next test
curl http://localhost:30089/stats/reset
```

## Test Matrix Configuration

The driver executes a comprehensive test matrix matching `run_all_locust_tests.ps1`:

```python
replica_counts = [1, 2, 4, 8]      # Inference pods
worker_counts = [4, 8]              # Locust workers
user_counts = [200, 400, 800]       # Virtual users
test_duration = 120                 # Seconds per test
cooldown = 15                       # Seconds between tests

# Total scenarios: 4 × 2 × 3 = 24 tests
# Estimated time: (120 + 15) × 24 = 54 minutes
```

### Customizing Test Matrix

Edit `locust-driver.py` and modify the `main()` function:

```python
# Quick smoke test (3 scenarios, ~7 minutes)
replica_counts = [2]
worker_counts = [4]
user_counts = [200, 400, 800]

# Production baseline (6 scenarios, ~14 minutes)
replica_counts = [2, 4]
worker_counts = [4, 8]
user_counts = [400]

# Full matrix (24 scenarios, ~54 minutes)
replica_counts = [1, 2, 4, 8]
worker_counts = [4, 8]
user_counts = [200, 400, 800]
```

Or use environment variables in the Job:

```yaml
env:
- name: REPLICA_COUNTS
  value: "2,4,8"
- name: WORKER_COUNTS
  value: "4,8"
- name: USER_COUNTS
  value: "200,400"
- name: TEST_DURATION
  value: "60"  # Shorter tests
```

## Results Format

### CSV Output (auto_summary.csv)

```csv
Replicas,Workers,Users,RPS,Median_ms,P95_ms,P99_ms,Failures_Pct,Total_Requests,Duration_s,Timestamp,Status
1,4,200,12.50,850,2500,3200,0.00,1500,120,2024-01-15T10:30:00,SUCCESS
2,4,200,18.75,620,1800,2400,0.00,2250,120,2024-01-15T10:32:30,SUCCESS
...
```

### Markdown Output (auto_summary.md)

Includes:
- Test execution metadata
- Configuration matrix table with all results
- Key findings (best throughput, best latency, averages)
- Recommendations for high-performance configurations
- Latency optimization suggestions

Example:

```markdown
# Kubernetes Locust Load Test Results

**Test Execution:** 2024-01-15 10:30:00
**Test Duration per Scenario:** 120s
**Total Scenarios:** 24

## Test Matrix

| Replicas | Workers | Users | RPS | Median (ms) | P95 (ms) | ...
|----------|---------|-------|-----|-------------|----------|
| 1        | 4       | 200   | 12.50 | 850       | 2500     | ...
| 2        | 4       | 200   | 18.75 | 620       | 1800     | ...
...

## Key Findings

- **Best Throughput:** 21.23 RPS with 4 replicas, 8 workers, 800 users
- **Best Latency:** 420ms median with 8 replicas, 4 workers, 200 users
- **Average Throughput:** 17.45 RPS
- **Success Rate:** 24/24 scenarios

## Recommendations

**High-Performance Configurations:**
- 4 replicas × 8 workers × 800 users → 21.23 RPS, 1200ms median
- 8 replicas × 8 workers × 400 users → 19.87 RPS, 650ms median
...
```

## Monitoring and Debugging

### Check Locust Master Status

```bash
# Web UI (if using NodePort)
open http://localhost:30089

# API health check
curl http://localhost:30089/

# Get current stats
curl http://localhost:30089/stats/requests | jq
```

### Monitor Worker Connectivity

```bash
# Check worker logs
kubectl logs -l app=locust,role=worker --tail=50

# Expected output:
# [2024-01-15 10:30:00] Locust worker connected to master at locust-master:5557
```

### Monitor Driver Progress

```bash
# Follow job logs
kubectl logs -f job/locust-load-test-matrix

# Check job status
kubectl get job locust-load-test-matrix

# Describe job for events
kubectl describe job locust-load-test-matrix
```

### Debug Scaling Issues

```bash
# Check deployment status
kubectl get deployment inference locust-worker

# Check pod readiness
kubectl get pods -l app=inference -o wide
kubectl get pods -l app=locust,role=worker -o wide

# View recent events
kubectl get events --sort-by='.lastTimestamp' | tail -20
```

### Common Issues

**Issue: Workers not connecting to master**
```bash
# Check service DNS resolution from worker pod
kubectl exec -it $(kubectl get pod -l app=locust,role=worker -o jsonpath='{.items[0].metadata.name}') -- nslookup locust-master

# Check master logs
kubectl logs -l app=locust,role=master
```

**Issue: Driver cannot scale deployments**
```bash
# Verify RBAC permissions
kubectl auth can-i update deployments --as=system:serviceaccount:default:locust-driver

# Check ServiceAccount exists
kubectl get serviceaccount locust-driver
kubectl get role locust-driver-role
kubectl get rolebinding locust-driver-rolebinding
```

**Issue: Locust API not responding**
```bash
# Test master endpoint
kubectl run curl-test --image=curlimages/curl:latest --rm -it --restart=Never -- \
  curl -v http://locust-master:8089/

# Check master pod health
kubectl get pod -l app=locust,role=master
kubectl logs -l app=locust,role=master --tail=100
```

## Performance Tuning

### Inference Resource Limits

Based on `LOAD_TEST_RESULTS.md` findings:

```yaml
# Current limits (sufficient for 10-50 users)
resources:
  requests:
    memory: "1Gi"
    cpu: "1000m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

# Recommended for 50-100 users
resources:
  requests:
    memory: "2Gi"
    cpu: "2000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Worker Resource Allocation

```yaml
# Light load (200 users total)
resources:
  requests:
    memory: "128Mi"
    cpu: "100m"
  limits:
    memory: "256Mi"
    cpu: "250m"

# Heavy load (800 users total)
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### Cluster Sizing Recommendations

For full 24-test matrix execution:

```
Minimum Cluster:
- 3 nodes × 4GB RAM × 2 vCPU
- Total: 12GB RAM, 6 vCPU
- Supports: 8 inference replicas + 8 workers + overhead

Recommended Cluster:
- 4 nodes × 8GB RAM × 4 vCPU
- Total: 32GB RAM, 16 vCPU
- Supports: Comfortable headroom for peak load

Docker Desktop (Development):
- 1 node × 6GB RAM × 4 vCPU
- Supports: Up to 4 inference replicas + 4 workers
- Limit test matrix: replica_counts = [1, 2], worker_counts = [4]
```

## Comparing with PowerShell Script

| Feature | PowerShell (run_all_locust_tests.ps1) | Kubernetes (This Implementation) |
|---------|---------------------------------------|----------------------------------|
| **Scaling** | `docker compose up -d --scale` | `kubectl scale deployment` |
| **Test Matrix** | 24 scenarios (1,2,4,8 × 4,8 × 200,400,800) | ✅ Identical |
| **Live Monitoring** | Progress bar with RPS/latency | ✅ Console output with stats |
| **Results** | CSV + Markdown | ✅ CSV + Markdown |
| **API Endpoints** | POST /swarm, GET /stats/requests, /stop, /stats/reset | ✅ Same |
| **Duration** | 120s per test | ✅ Configurable (default 120s) |
| **Cooldown** | 15s between tests | ✅ Configurable (default 15s) |
| **Automation** | PowerShell script | Python script in Kubernetes Job |
| **Portability** | Windows + Docker Compose | ✅ Any Kubernetes cluster |
| **CI/CD Integration** | Limited | ✅ Native (Job/CronJob) |

## Integration with CI/CD

### GitLab CI Example

```yaml
load-test:
  stage: test
  script:
    - kubectl apply -f .k8s/locust-configmap.yaml
    - kubectl apply -f .k8s/locust-master.yaml
    - kubectl apply -f .k8s/locust-worker.yaml
    - kubectl wait --for=condition=ready pod -l app=locust --timeout=120s
    - kubectl delete job locust-load-test-matrix --ignore-not-found
    - kubectl create job locust-load-test-matrix --image=locust-driver:latest
    - kubectl wait --for=condition=complete job/locust-load-test-matrix --timeout=60m
    - POD=$(kubectl get pod -l job-name=locust-load-test-matrix -o jsonpath='{.items[0].metadata.name}')
    - kubectl cp $POD:/results/auto_summary.csv ./auto_summary.csv
    - kubectl cp $POD:/results/auto_summary.md ./auto_summary.md
  artifacts:
    paths:
      - auto_summary.csv
      - auto_summary.md
    expire_in: 30 days
  only:
    - main
```

### GitHub Actions Example

```yaml
name: Load Test
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday 2 AM

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: azure/k8s-set-context@v3
      with:
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
    - name: Deploy Locust
      run: |
        kubectl apply -f .k8s/locust-configmap.yaml
        kubectl apply -f .k8s/locust-master.yaml
        kubectl apply -f .k8s/locust-worker.yaml
        kubectl wait --for=condition=ready pod -l app=locust --timeout=120s
    - name: Run Load Test
      run: |
        kubectl delete job locust-load-test-matrix --ignore-not-found
        kubectl apply -f .k8s/locust-driver-job.yaml
        kubectl wait --for=condition=complete job/locust-load-test-matrix --timeout=60m
    - name: Collect Results
      run: |
        POD=$(kubectl get pod -l job-name=locust-load-test-matrix -o jsonpath='{.items[0].metadata.name}')
        kubectl cp $POD:/results/auto_summary.csv ./auto_summary.csv
        kubectl cp $POD:/results/auto_summary.md ./auto_summary.md
    - name: Upload Artifacts
      uses: actions/upload-artifact@v3
      with:
        name: load-test-results
        path: |
          auto_summary.csv
          auto_summary.md
```

## Scheduled Testing with CronJob

For regular performance regression testing:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: weekly-load-test
  namespace: default
spec:
  schedule: "0 2 * * 0"  # Every Sunday at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: locust-driver
          restartPolicy: OnFailure
          containers:
          - name: driver
            image: locust-driver:latest
            volumeMounts:
            - name: results
              mountPath: /results
          volumes:
          - name: results
            persistentVolumeClaim:
              claimName: locust-results-pvc
```

## Cleanup

```bash
# Remove all Locust resources
kubectl delete deployment locust-master locust-worker
kubectl delete service locust-master
kubectl delete configmap locust-scripts locust-driver-script
kubectl delete job locust-load-test-matrix
kubectl delete serviceaccount locust-driver
kubectl delete role locust-driver-role
kubectl delete rolebinding locust-driver-rolebinding

# Or use labels
kubectl delete all,cm,sa,role,rolebinding -l app=locust
```

## Next Steps

1. **Grafana Integration**: Visualize Locust metrics in real-time
   - Deploy Prometheus to scrape Locust `/metrics` endpoint
   - Import Locust dashboard to Grafana
   - Set up alerts for failure rate or latency thresholds

2. **Advanced Scenarios**: Extend locustfile.py with more complex flows
   - Multi-step transactions (download → preprocess → predict)
   - Authentication flows
   - WebSocket testing

3. **Cloud Deployment**: Deploy to production Kubernetes
   - Use managed Kubernetes (EKS, GKE, AKS)
   - Configure Ingress for Locust master
   - Store results in S3/GCS/Azure Blob

4. **Performance Baselines**: Establish SLAs and alerts
   - Define acceptable latency: p95 < 1000ms
   - Define minimum throughput: RPS > 15
   - Alert on regression: >20% latency increase

## Reference Links

- **Locust Documentation**: https://docs.locust.io/
- **Kubernetes Jobs**: https://kubernetes.io/docs/concepts/workloads/controllers/job/
- **RBAC Authorization**: https://kubernetes.io/docs/reference/access-authn-authz/rbac/
- **FLTS Production Guide**: `FLTS_PRODUCTION_READINESS_REPORT.md`
- **Load Test Results**: `LOAD_TEST_RESULTS.md`
