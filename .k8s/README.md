# Kubernetes Locust Load Testing

**Distributed load testing for FLTS inference service - Kubernetes-native implementation**

This directory contains manifests and automation for running distributed Locust load tests natively in Kubernetes, replicating and enhancing the functionality of `run_all_locust_tests.ps1`.

## 🚀 Quick Start

### One-Command Deployment (Windows PowerShell)

```powershell
# Deploy infrastructure and run full 24-scenario test matrix (~54 minutes)
.\quick-start.ps1 -All

# Results will be available in: .\locust-results\
```

### Manual Deployment (Cross-Platform)

```bash
# 1. Deploy infrastructure
kubectl apply -f locust-configmap.yaml
kubectl apply -f locust-master.yaml
kubectl apply -f locust-worker.yaml

# 2. Wait for ready state
kubectl wait --for=condition=ready pod -l app=locust --timeout=120s

# 3. Build driver image
docker build -t locust-driver:latest -f Dockerfile.driver .

# 4. Run test matrix
kubectl apply -f locust-driver-job-simple.yaml

# 5. Monitor progress
kubectl logs -f job/locust-load-test-matrix

# 6. Retrieve results
POD=$(kubectl get pod -l job-name=locust-load-test-matrix -o jsonpath='{.items[0].metadata.name}')
kubectl cp $POD:/results/auto_summary.csv ./auto_summary.csv
kubectl cp $POD:/results/auto_summary.md ./auto_summary.md
```

## 📁 Files Overview

| File | Purpose | Required |
|------|---------|----------|
| **locust-configmap.yaml** | Test script (simplified 60-line locustfile.py) | ✅ Yes |
| **locust-master.yaml** | Master deployment + NodePort service (30089) | ✅ Yes |
| **locust-worker.yaml** | Worker deployment (scalable 4-8 replicas) | ✅ Yes |
| **locust-driver.py** | Python orchestration script (550 lines) | ✅ Yes |
| **locust-driver-job.yaml** | Job with RBAC (complex, reference) | ⚠️ Optional |
| **locust-driver-job-simple.yaml** | Simplified Job (uses custom image) | ✅ Recommended |
| **Dockerfile.driver** | Custom driver image (Python + kubectl) | ✅ Recommended |
| **quick-start.ps1** | PowerShell automation script | ⚠️ Windows only |
| **K8S_LOCUST_DEPLOYMENT_GUIDE.md** | Complete deployment guide (650+ lines) | 📖 Reference |
| **IMPLEMENTATION_SUMMARY.md** | Feature comparison and summary | 📖 Reference |
| **ARCHITECTURE_DIAGRAMS.md** | Visual architecture diagrams | 📖 Reference |
| **README.md** | This file | 📖 You are here |

## 🧪 Test Matrix

The driver executes a comprehensive test matrix matching the original PowerShell script:

```
Inference Replicas: [1, 2, 4, 8]
Locust Workers: [4, 8]
Virtual Users: [200, 400, 800]

Total Scenarios: 4 × 2 × 3 = 24 tests
Duration per Test: 120 seconds
Cooldown: 15 seconds
Estimated Total Time: ~54 minutes
```

### Customizing the Test Matrix

Edit `locust-driver.py` line 466-468:

```python
# Quick smoke test (3 scenarios, ~7 minutes)
replica_counts = [2]
worker_counts = [4]
user_counts = [200, 400, 800]

# Production baseline (6 scenarios, ~14 minutes)
replica_counts = [2, 4]
worker_counts = [4, 8]
user_counts = [400]

# Full matrix (24 scenarios, ~54 minutes) - DEFAULT
replica_counts = [1, 2, 4, 8]
worker_counts = [4, 8]
user_counts = [200, 400, 800]
```

## 📊 Results

The driver generates two output files:

### CSV Report (`auto_summary.csv`)

```csv
Replicas,Workers,Users,RPS,Median_ms,P95_ms,P99_ms,Failures_Pct,Total_Requests,Duration_s,Timestamp,Status
1,4,200,12.50,850,2500,3200,0.00,1500,120,2024-01-15T10:30:00,SUCCESS
2,4,200,18.75,620,1800,2400,0.00,2250,120,2024-01-15T10:32:30,SUCCESS
...
```

### Markdown Report (`auto_summary.md`)

Includes:
- Test execution metadata
- Configuration matrix table (all 24 scenarios)
- Key findings (best throughput, best latency, averages)
- Performance recommendations
- Latency optimization suggestions

## 🔧 Prerequisites

### Required

- **Kubernetes Cluster**: Docker Desktop (6GB+ RAM), Minikube, or production cluster (EKS/GKE/AKS)
- **kubectl**: Configured with cluster access
- **Inference Deployment**: Must be running in `default` namespace

### Optional

- **Docker**: For building custom driver image (recommended)
- **Python 3.11+**: For running driver script locally
- **PowerShell**: For quick-start automation (Windows)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Locust Master (NodePort 30089)                         │
│    → Web UI: http://localhost:30089                     │
│    → API: /swarm, /stats/requests, /stop               │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   Worker 1     Worker 2     Worker 3 ...
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Inference Service     │
        │  http://inference:8000 │
        │  (1-8 replicas)        │
        └────────────────────────┘

   Driver Job (orchestration):
   1. Scale inference (kubectl)
   2. Scale workers (kubectl)
   3. Start test (Locust API)
   4. Poll stats (live monitoring)
   5. Export results (CSV + MD)
```

See [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) for detailed diagrams.

## 📖 Documentation

- **[K8S_LOCUST_DEPLOYMENT_GUIDE.md](K8S_LOCUST_DEPLOYMENT_GUIDE.md)**: Complete deployment guide with troubleshooting
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Feature comparison with PowerShell script
- **[ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)**: Visual architecture and flow diagrams

## 🔍 Monitoring

### Check Locust Status

```bash
# Web UI (if using NodePort)
open http://localhost:30089

# API health check
curl http://localhost:30089/

# Get current stats
curl http://localhost:30089/stats/requests | jq '.stats[] | select(.name=="Aggregated")'
```

### Monitor Worker Connectivity

```bash
# Check worker logs
kubectl logs -l app=locust,role=worker --tail=50

# Expected: "Locust worker connected to master at locust-master:5557"
```

### Monitor Driver Progress

```bash
# Follow job logs
kubectl logs -f job/locust-load-test-matrix

# Check job status
kubectl get job locust-load-test-matrix

# Get results (when complete)
POD=$(kubectl get pod -l job-name=locust-load-test-matrix -o jsonpath='{.items[0].metadata.name}')
kubectl logs $POD --tail=50
```

## 🛠️ Troubleshooting

### Workers Not Connecting

```bash
# Check service DNS resolution
kubectl exec -it <worker-pod> -- nslookup locust-master

# Check master logs
kubectl logs -l app=locust,role=master
```

### Driver Cannot Scale Deployments

```bash
# Verify RBAC permissions
kubectl auth can-i update deployments --as=system:serviceaccount:default:locust-driver

# Check ServiceAccount and Role
kubectl get serviceaccount locust-driver
kubectl get role locust-driver-role
kubectl get rolebinding locust-driver-rolebinding
```

### Locust API Not Responding

```bash
# Test master endpoint from within cluster
kubectl run curl-test --image=curlimages/curl:latest --rm -it --restart=Never -- \
  curl -v http://locust-master:8089/

# Check master pod health
kubectl get pod -l app=locust,role=master
kubectl logs -l app=locust,role=master --tail=100
```

See [K8S_LOCUST_DEPLOYMENT_GUIDE.md](K8S_LOCUST_DEPLOYMENT_GUIDE.md) for more troubleshooting steps.

## 🚦 Performance Expectations

Based on `LOAD_TEST_RESULTS.md` baseline with 2 inference replicas:

| Users | RPS | P95 Latency | Status |
|-------|-----|-------------|--------|
| ≤10 | 16.4 | 686ms | ✅ Optimal |
| 11-50 | 16.8 | 3,930ms | ⚠️ Degraded (473% increase) |
| 50-100 | 21.2 | 5,463ms | 🔴 Saturated (bottleneck) |

**Recommendations**:
- **50 users**: 5-10 inference replicas
- **100 users**: 8-12 inference replicas + GPU acceleration
- **200+ users**: 10-15 replicas + horizontal autoscaling

## 🎯 Use Cases

### Development Testing

```bash
# Quick smoke test (single configuration)
kubectl scale deployment inference --replicas=2
kubectl scale deployment locust-worker --replicas=4

curl -X POST http://localhost:30089/swarm \
  -d "user_count=200&spawn_rate=20&host=http://inference:8000"

# Monitor for 2 minutes, then stop
sleep 120
curl http://localhost:30089/stop
```

### CI/CD Integration

```yaml
# GitLab CI example
load-test:
  stage: test
  script:
    - kubectl apply -f .k8s/
    - kubectl wait --for=condition=complete job/locust-load-test-matrix --timeout=60m
    - kubectl cp $(kubectl get pod -l job-name=locust-load-test-matrix -o jsonpath='{.items[0].metadata.name}'):/results/ ./
  artifacts:
    paths:
      - auto_summary.csv
      - auto_summary.md
```

### Scheduled Performance Testing

```yaml
# CronJob for weekly testing
apiVersion: batch/v1
kind: CronJob
metadata:
  name: weekly-load-test
spec:
  schedule: "0 2 * * 0"  # Every Sunday 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          # ... (same as locust-driver-job-simple.yaml)
```

## 🧹 Cleanup

```bash
# Remove all Locust resources
kubectl delete all,cm,sa,role,rolebinding -l app=locust

# Or use quick-start script (Windows)
.\quick-start.ps1 -Cleanup
```

## 🔄 Comparison with PowerShell Script

| Aspect | PowerShell (run_all_locust_tests.ps1) | Kubernetes (This Implementation) |
|--------|---------------------------------------|----------------------------------|
| **Platform** | Windows + Docker Compose | Any Kubernetes cluster |
| **Scaling** | `docker compose --scale` | `kubectl scale deployment` |
| **Test Matrix** | 24 scenarios | ✅ Identical |
| **Live Monitoring** | Progress bar + RPS | ✅ Console stats |
| **Results** | CSV + Markdown | ✅ CSV + Markdown |
| **Automation** | PowerShell script | Python + Kubernetes Job |
| **CI/CD** | Limited | ✅ Native (Job/CronJob) |
| **Portability** | Windows only | ✅ Cross-platform |

## 📚 Additional Resources

- **Locust Documentation**: https://docs.locust.io/
- **Kubernetes Jobs**: https://kubernetes.io/docs/concepts/workloads/controllers/job/
- **FLTS Production Guide**: `../FLTS_PRODUCTION_READINESS_REPORT.md`
- **Load Test Results**: `../LOAD_TEST_RESULTS.md`

## 🤝 Contributing

When adding new test scenarios or modifying the test matrix:

1. Update `locust-driver.py` (main test logic)
2. Update `IMPLEMENTATION_SUMMARY.md` (feature documentation)
3. Update this README if new files are added
4. Test changes locally before committing

## 📝 License

Part of the FLTS (Federated Learning Time Series) project.

---

**Version**: 1.0  
**Last Updated**: January 2024  
**Maintainer**: FLTS Team
