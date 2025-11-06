# Kubernetes Locust Implementation - Summary

## Overview

Successfully implemented **Kubernetes-native distributed Locust load testing** to replicate and enhance the functionality of `run_all_locust_tests.ps1`. The solution provides automated test matrix execution, live monitoring, and comprehensive reporting - all running natively in Kubernetes.

## Files Created

### Core Kubernetes Manifests

1. **`.k8s/locust-configmap.yaml`**
   - ConfigMap containing simplified Locust test script (60 lines vs original 761)
   - Tasks: `predict` (80%), `health_check` (10%), `metrics_check` (10%)
   - Payload: 30 timestamps × 11 features matching FLTS requirements
   - **Purpose**: Provides test script to master and worker pods

2. **`.k8s/locust-master.yaml`**
   - Deployment: 1 replica, locustio/locust:latest
   - Environment: LOCUST_MODE=master, TARGET_HOST=http://inference:8000
   - Service: NodePort 30089 (web UI + API access)
   - Ports: 8089 (HTTP), 5557 (master-p1), 5558 (master-p2)
   - Resources: 512Mi-1Gi RAM, 500m-1000m CPU
   - **Purpose**: Coordinates distributed testing, exposes API for orchestration

3. **`.k8s/locust-worker.yaml`**
   - Deployment: 4 replicas (scalable to 8+ via kubectl)
   - Environment: LOCUST_MODE=worker, LOCUST_MASTER_NODE_HOST=locust-master
   - Resources: 256Mi-512Mi RAM, 250m-500m CPU per worker
   - **Purpose**: Executes load tests, connects to master for coordination

4. **`.k8s/locust-driver-job.yaml`**
   - Kubernetes Job with ServiceAccount and RBAC (scale deployments permission)
   - Role: Allows updating deployments and deployments/scale in default namespace
   - RoleBinding: Attaches role to locust-driver ServiceAccount
   - **Purpose**: Provides RBAC permissions for driver automation

5. **`.k8s/locust-driver-job-simple.yaml`**
   - Simplified Job using custom driver image
   - Environment variables: LOCUST_MASTER_URL, test duration, cooldown
   - Volume: emptyDir for results storage
   - **Purpose**: Clean Job definition for production use

### Automation and Tooling

6. **`.k8s/locust-driver.py`** (550 lines)
   - **Class**: `LoadTestDriver` - Orchestrates full test matrix
   - **Functions**:
     - `scale_deployment()`: Uses kubectl to scale inference and workers
     - `start_locust_test()`: POST to /swarm API endpoint
     - `get_locust_stats()`: GET from /stats/requests API
     - `stop_locust_test()`: Stop via /stop endpoint
     - `reset_locust_stats()`: Reset via /stats/reset endpoint
     - `poll_test_with_progress()`: Live monitoring with RPS/latency display
     - `run_test_scenario()`: Execute single test configuration
     - `run_test_matrix()`: Loop through all 24 scenarios
     - `export_results_csv()`: Generate CSV report
     - `export_results_markdown()`: Generate MD report with analysis
   - **Test Matrix**: 4 replica counts × 2 worker counts × 3 user levels = 24 tests
   - **Purpose**: Replicates PowerShell script functionality in Python

7. **`.k8s/Dockerfile.driver`**
   - Base: python:3.11-slim
   - Includes: kubectl binary, requests library
   - Entrypoint: locust-driver.py
   - **Purpose**: Custom image for Job execution (recommended approach)

8. **`.k8s/quick-start.ps1`** (PowerShell automation)
   - **Commands**:
     - `-All`: Deploy + Run full test matrix
     - `-Deploy`: Deploy Locust infrastructure only
     - `-Run`: Start load test job (builds image if needed)
     - `-Monitor`: Follow job logs in real-time
     - `-Results`: Download CSV/MD results from pod
     - `-Cleanup`: Remove all resources
   - **Purpose**: One-command deployment and execution for Windows users

9. **`.k8s/K8S_LOCUST_DEPLOYMENT_GUIDE.md`** (650+ lines)
   - **Sections**:
     - Architecture overview with diagrams
     - 3 deployment options (custom image, local script, manual)
     - Test matrix configuration and customization
     - Results format (CSV + Markdown examples)
     - Monitoring and debugging commands
     - Performance tuning recommendations
     - Common issues and solutions
     - CI/CD integration (GitLab, GitHub Actions)
     - CronJob example for scheduled testing
   - **Purpose**: Complete reference documentation

## Feature Parity with PowerShell Script

| Feature | PowerShell (run_all_locust_tests.ps1) | Kubernetes Implementation | Status |
|---------|---------------------------------------|---------------------------|--------|
| Distributed Locust (master + workers) | ✅ Docker Compose | ✅ Kubernetes Deployments | ✅ |
| Test Matrix (24 scenarios) | ✅ 1,2,4,8 × 4,8 × 200,400,800 | ✅ Identical | ✅ |
| Automated Scaling | ✅ docker compose --scale | ✅ kubectl scale deployment | ✅ |
| Locust API Usage | ✅ POST /swarm, GET /stats | ✅ Same endpoints | ✅ |
| Live Progress Monitoring | ✅ Progress bar + RPS display | ✅ Console output with stats | ✅ |
| Results: CSV | ✅ auto_summary.csv | ✅ auto_summary.csv | ✅ |
| Results: Markdown | ✅ auto_summary.md | ✅ auto_summary.md + analysis | ✅ |
| Test Duration (120s) | ✅ Configurable | ✅ Configurable (env var) | ✅ |
| Cooldown (15s) | ✅ Between tests | ✅ Between tests | ✅ |
| Error Handling | ✅ Try/Catch blocks | ✅ Exception handling | ✅ |
| Success/Failure Tracking | ✅ Status column | ✅ Status + ERROR messages | ✅ |
| Best Performance Analysis | ✅ Manual review | ✅ Automated in MD report | 🆕 Enhanced |
| Portability | ❌ Windows + Docker only | ✅ Any Kubernetes cluster | 🆕 Improvement |
| CI/CD Integration | ⚠️ Limited | ✅ Native Job/CronJob | 🆕 Improvement |

## Test Matrix Configuration

```python
# Default configuration (matching PowerShell script)
replica_counts = [1, 2, 4, 8]      # Inference deployment replicas
worker_counts = [4, 8]              # Locust worker count
user_counts = [200, 400, 800]       # Virtual users per test
test_duration = 120                 # Seconds per scenario
cooldown = 15                       # Seconds between scenarios

# Total scenarios: 4 × 2 × 3 = 24 tests
# Estimated time: (120 + 15) × 24 = 54 minutes
```

### Customization Options

**Quick Smoke Test** (3 scenarios, ~7 minutes):
```python
replica_counts = [2]
worker_counts = [4]
user_counts = [200, 400, 800]
```

**Production Baseline** (6 scenarios, ~14 minutes):
```python
replica_counts = [2, 4]
worker_counts = [4, 8]
user_counts = [400]
```

## Usage Examples

### Quick Start (One Command)

```powershell
# Deploy everything and run full test matrix
.\quick-start.ps1 -All

# Results available in ~54 minutes at:
# - .\locust-results\auto_summary.csv
# - .\locust-results\auto_summary.md
```

### Manual Deployment (Step-by-Step)

```bash
# 1. Deploy Locust infrastructure
kubectl apply -f .k8s/locust-configmap.yaml
kubectl apply -f .k8s/locust-master.yaml
kubectl apply -f .k8s/locust-worker.yaml

# 2. Wait for ready state
kubectl wait --for=condition=ready pod -l app=locust --timeout=120s

# 3. Verify master is accessible
curl http://localhost:30089/

# 4. Build driver image
cd .k8s
docker build -t locust-driver:latest -f Dockerfile.driver .

# 5. Run test matrix
kubectl apply -f locust-driver-job-simple.yaml

# 6. Monitor progress
kubectl logs -f job/locust-load-test-matrix

# 7. Retrieve results (when complete)
POD=$(kubectl get pod -l job-name=locust-load-test-matrix -o jsonpath='{.items[0].metadata.name}')
kubectl cp $POD:/results/auto_summary.csv ./auto_summary.csv
kubectl cp $POD:/results/auto_summary.md ./auto_summary.md
```

### Local Development (No Job)

```bash
# 1. Deploy Locust only
kubectl apply -f .k8s/locust-configmap.yaml
kubectl apply -f .k8s/locust-master.yaml
kubectl apply -f .k8s/locust-worker.yaml

# 2. Port-forward master (optional)
kubectl port-forward svc/locust-master 8089:8089 &

# 3. Install dependencies locally
pip install requests

# 4. Run driver script directly
cd .k8s
python locust-driver.py
# Results saved to ./results/
```

## Results Format

### CSV Output (auto_summary.csv)

```csv
Replicas,Workers,Users,RPS,Median_ms,P95_ms,P99_ms,Failures_Pct,Total_Requests,Duration_s,Timestamp,Status
1,4,200,12.50,850,2500,3200,0.00,1500,120,2024-01-15T10:30:00,SUCCESS
2,4,200,18.75,620,1800,2400,0.00,2250,120,2024-01-15T10:32:30,SUCCESS
2,4,400,17.50,1200,3500,4200,0.50,2100,120,2024-01-15T10:35:00,SUCCESS
...
```

### Markdown Output (auto_summary.md)

Key sections included:
- **Test Matrix Table**: All 24 scenarios with RPS, latency, failures
- **Key Findings**: Best throughput, best latency, averages, success rate
- **Recommendations**: High-performance configurations, latency optimization tips

Example key findings:
```markdown
## Key Findings

- **Best Throughput:** 21.23 RPS with 4 replicas, 8 workers, 800 users
- **Best Latency:** 420ms median with 8 replicas, 4 workers, 200 users
- **Average Throughput:** 17.45 RPS
- **Average Median Latency:** 890ms
- **Success Rate:** 24/24 scenarios
```

## Monitoring and Debugging

### Check Deployment Status

```bash
# Verify all components
kubectl get deployments,pods,services -l app=locust

# Check worker connectivity
kubectl logs -l app=locust,role=worker --tail=20 | grep "connected to master"

# Test Locust API
curl http://localhost:30089/stats/requests | jq '.stats[] | select(.name=="Aggregated")'
```

### Monitor Job Progress

```bash
# Follow job logs
kubectl logs -f job/locust-load-test-matrix

# Check job status
kubectl get job locust-load-test-matrix
# NAME                       COMPLETIONS   DURATION   AGE
# locust-load-test-matrix   0/1           10m        10m

# Describe for events
kubectl describe job locust-load-test-matrix
```

### Debug Scaling Issues

```bash
# Check deployment scaling
kubectl get deployment inference locust-worker -o wide

# Verify RBAC permissions
kubectl auth can-i update deployments --as=system:serviceaccount:default:locust-driver

# View scaling events
kubectl get events --sort-by='.lastTimestamp' | grep -i scale
```

## Performance Considerations

Based on `LOAD_TEST_RESULTS.md` findings:

### Cluster Sizing

**Minimum (Docker Desktop Development)**:
- 1 node × 6GB RAM × 4 vCPU
- Supports: 4 inference replicas + 4 workers
- Limit: `replica_counts = [1, 2]`, `worker_counts = [4]`

**Recommended (Full Testing)**:
- 3-4 nodes × 4-8GB RAM × 2-4 vCPU each
- Total: 12-32GB RAM, 6-16 vCPU
- Supports: Full 24-test matrix

### Resource Allocation

**Inference Pods**:
```yaml
resources:
  requests:
    memory: "1Gi"    # 2Gi for >50 users
    cpu: "1000m"     # 2000m for >50 users
  limits:
    memory: "2Gi"    # 4Gi for >100 users
    cpu: "1000m"     # 2000m for >100 users
```

**Locust Workers**:
```yaml
resources:
  requests:
    memory: "256Mi"  # Per worker
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### Expected Performance

From `stress_test.py` baseline (2 inference replicas):
- **10 users**: 16.4 RPS, p95 686ms (optimal)
- **50 users**: 16.8 RPS, p95 3,930ms (degraded, 473% latency increase)
- **100 users**: 21.2 RPS, p95 5,463ms (saturated, bottleneck at 32-worker limit)

With optimized scaling (4-8 replicas):
- **Expected**: 30-40 RPS sustained throughput
- **Target**: p95 < 1000ms at 200-400 users
- **Recommendation**: 5-10 replicas for 50-100 concurrent users

## CI/CD Integration

### GitHub Actions (Production Example)

```yaml
name: Weekly Load Test
on:
  schedule:
    - cron: '0 2 * * 0'  # Every Sunday 2 AM

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: azure/k8s-set-context@v3
      with:
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
    - name: Deploy and Test
      run: |
        kubectl apply -f .k8s/locust-configmap.yaml
        kubectl apply -f .k8s/locust-master.yaml
        kubectl apply -f .k8s/locust-worker.yaml
        kubectl wait --for=condition=ready pod -l app=locust --timeout=120s
        kubectl apply -f .k8s/locust-driver-job-simple.yaml
        kubectl wait --for=condition=complete job/locust-load-test-matrix --timeout=60m
    - name: Collect Results
      run: |
        POD=$(kubectl get pod -l job-name=locust-load-test-matrix -o jsonpath='{.items[0].metadata.name}')
        kubectl cp $POD:/results/auto_summary.csv ./auto_summary.csv
        kubectl cp $POD:/results/auto_summary.md ./auto_summary.md
    - uses: actions/upload-artifact@v3
      with:
        name: load-test-results
        path: |
          auto_summary.csv
          auto_summary.md
```

### CronJob (Scheduled Testing)

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: weekly-load-test
spec:
  schedule: "0 2 * * 0"
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: locust-driver
          containers:
          - name: driver
            image: locust-driver:latest
            # ... (same config as Job)
```

## Advantages Over PowerShell Implementation

1. **Portability**: Runs on any Kubernetes cluster (Linux, Windows, macOS, cloud)
2. **CI/CD Native**: Job/CronJob primitives for automation
3. **Scalability**: Leverage cluster autoscaling for large tests
4. **Observability**: Native Prometheus metrics, pod logs, events
5. **Resource Management**: Requests/limits for predictable performance
6. **RBAC**: Fine-grained permissions for security
7. **Declarative**: GitOps-friendly YAML manifests
8. **Persistent Results**: Option to use PersistentVolumes instead of emptyDir
9. **Multi-tenancy**: Namespace isolation for parallel testing

## Known Limitations

1. **Locust API**: Stats endpoint often returns p90 instead of p95 (Locust limitation)
2. **Docker Desktop**: Limited to ~4 inference replicas + 4 workers (6GB RAM constraint)
3. **Results Persistence**: Default uses emptyDir (lost on pod deletion)
   - **Solution**: Use PersistentVolumeClaim or copy results before Job cleanup
4. **Job TTL**: Results available for 1 hour after completion (ttlSecondsAfterFinished: 3600)
   - **Solution**: Copy results immediately or increase TTL

## Cleanup

```bash
# Remove all resources
kubectl delete all,cm,sa,role,rolebinding -l app=locust

# Or use quick-start script
.\quick-start.ps1 -Cleanup
```

## Next Steps

1. **Deploy to Production Cluster**: Use managed Kubernetes (EKS, GKE, AKS)
2. **Grafana Integration**: Visualize metrics in real-time
3. **Prometheus Alerts**: Alert on latency/failure thresholds
4. **PersistentVolume**: Store historical results
5. **Custom Metrics**: Export RPS/latency to custom monitoring system
6. **Advanced Scenarios**: Extend locustfile.py with authentication, multi-step flows

## References

- **Deployment Guide**: `.k8s/K8S_LOCUST_DEPLOYMENT_GUIDE.md` (650+ lines)
- **Load Test Results**: `LOAD_TEST_RESULTS.md` (baseline performance analysis)
- **Production Readiness**: `FLTS_PRODUCTION_READINESS_REPORT.md` (complete deployment guide)
- **PowerShell Reference**: `run_all_locust_tests.ps1` (original implementation)
- **Locust Docs**: https://docs.locust.io/
- **Kubernetes Jobs**: https://kubernetes.io/docs/concepts/workloads/controllers/job/

---

**Implementation Date**: January 2024  
**Status**: ✅ Complete - Ready for deployment and testing  
**Test Matrix**: 24 scenarios (4 replicas × 2 workers × 3 user levels)  
**Estimated Time**: 54 minutes for full matrix  
**Compatibility**: Kubernetes 1.20+, Docker Desktop, Minikube, EKS/GKE/AKS
