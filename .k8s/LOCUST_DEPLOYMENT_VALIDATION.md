# Locust Deployment Validation Report

**Date**: November 4, 2025  
**Cluster**: Docker Desktop Kubernetes  
**Namespace**: default

## Executive Summary

✅ **DEPLOYMENT SUCCESSFUL** - Locust distributed load testing stack is fully operational with master and 4 workers connected and ready for load testing.

## Deployment Steps Completed

### 1. ConfigMap Deployment
```bash
kubectl apply -f locust-configmap.yaml
```
**Status**: ✅ Success  
**ConfigMap**: `locust-scripts` created with simplified locustfile.py

### 2. Master Deployment
```bash
kubectl delete deployment locust-master  # Removed old Helm deployment
kubectl apply -f locust-master.yaml
```
**Status**: ✅ Success  
**Deployment**: `locust-master` with 1 replica
**Configuration**:
- Image: `locustio/locust:latest`
- Mode: `--master`
- Bind Host: `0.0.0.0`
- Bind Port: `5557`
- Expected Workers: `4`

### 3. Worker Deployment
```bash
kubectl apply -f locust-worker.yaml
```
**Status**: ✅ Success  
**Deployment**: `locust-worker` with 4 replicas
**Configuration**:
- Image: `locustio/locust:latest`
- Mode: `--worker`
- Master Host: `locust-master`
- Master Port: `5557`

### 4. Service Configuration
```bash
kubectl delete svc locust-master  # Removed old service with incorrect selector
kubectl apply -f locust-master.yaml
```
**Status**: ✅ Success  
**Issue Resolved**: Original service had Helm-managed labels (`app.kubernetes.io/instance: flts`) that didn't match our pod labels. Recreated service with correct selector: `app: locust, role: master`

## Current Infrastructure Status

### Pods
```
NAME                             READY   STATUS    RESTARTS   AGE
locust-master-65f66fb6cd-f4cqc   1/1     Running   0          3m45s
locust-worker-8cbbccc79-6bxq6    1/1     Running   0          3m45s
locust-worker-8cbbccc79-k4zcd    1/1     Running   0          3m43s
locust-worker-8cbbccc79-qf4f8    1/1     Running   0          3m42s
locust-worker-8cbbccc79-z5rpl    1/1     Running   0          3m45s
```
**Total Pods**: 5 (1 master + 4 workers)  
**Health**: All Running and Ready

### Service
```
NAME            TYPE       CLUSTER-IP       PORT(S)                                        
locust-master   NodePort   10.105.234.202   8089:30089/TCP,5557:32536/TCP,5558:32005/TCP
```
**Type**: NodePort  
**Web UI Port**: 30089 (external), 8089 (internal)  
**Master Ports**: 5557 (P1), 5558 (P2)

### Endpoints
```
NAME            ENDPOINTS                                         
locust-master   10.1.3.204:5557,10.1.3.204:8089,10.1.3.204:5558
```
**Status**: ✅ All ports exposed and routing correctly

## Worker Connectivity Validation

### Master Logs - Worker Connections
```
[2025-11-04 21:31:18,803] locust-master.../INFO/locust.runners: 
  locust-worker-8cbbccc79-k4zcd_... (index 2) reported as ready. 3 workers connected.

[2025-11-04 21:31:18,891] locust-master.../INFO/locust.runners: 
  locust-worker-8cbbccc79-6bxq6_... (index 3) reported as ready. 4 workers connected.
```

✅ **All 4 workers successfully connected to master**

### API Verification
```json
{
  "state": "ready",
  "workers": 4,
  "user_count": 0
}
```

## Smoke Test Results

### Test Configuration
- **Users**: 10
- **Spawn Rate**: 2 users/second
- **Target Host**: http://inference:8000
- **Duration**: 5 seconds

### Results
```
RPS: 2.0
Total Requests: 11
Failures: 8
Median Latency: 16ms
```

### Analysis
✅ **Locust System Working**: Test successfully initiated, workers generated requests  
⚠️ **High Failure Rate**: 72.7% (8/11 requests failed)  
**Root Cause**: Inference deployment not running in cluster

```bash
kubectl get pods -l app=inference
# No resources found in default namespace
```

**Recommendation**: Deploy inference service before running production load tests.

## Locust Web UI Access

### NodePort (Production)
```
http://localhost:30089
```
**Status**: ✅ Accessible  
**Features**: Full web UI with start/stop controls, real-time stats, charts

### Port-Forward (Development)
```bash
kubectl port-forward svc/locust-master 8089:8089
# Access at: http://localhost:8089
```

## API Endpoints Verified

### 1. Start Test
```bash
curl -X POST http://localhost:30089/swarm \
  -d "user_count=10&spawn_rate=2&host=http://inference:8000"
```
**Response**: 
```json
{
  "host": "http://inference:8000",
  "message": "Swarming started",
  "success": true
}
```
✅ Working

### 2. Get Stats
```bash
curl http://localhost:30089/stats/requests
```
**Response**: JSON with `stats`, `workers`, `user_count`, `state`  
✅ Working

### 3. Stop Test
```bash
curl http://localhost:30089/stop
```
✅ Working

### 4. Reset Stats
```bash
curl http://localhost:30089/stats/reset
```
✅ Working

## Configuration Files

### Locust Test Script
**Location**: ConfigMap `locust-scripts` → `/home/locust/locustfile.py`  
**Test Class**: `InferenceUser`  
**Tasks**:
- `@task(8)` predict(): POST /predict (80% weight)
- `@task(1)` health_check(): GET /healthz (10% weight)
- `@task(1)` metrics_check(): GET /metrics (10% weight)

**Payload Structure** (30 timestamps × 11 features):
```python
{
  "data": {
    "ts": ["2018-02-06 00:00:00"] * 30,
    "down": [109934672.0] * 30,
    "up": [41703548.0] * 30,
    "rnti_count": [96.0] * 30,
    "mcs_down": [18.52] * 30,
    # ... 6 more features
  }
}
```

## Troubleshooting Issues Encountered

### Issue 1: Workers Not Connecting
**Symptom**: Workers continuously retrying connection to master
```
[INFO] Failed to connect to master locust-master:5557, retry 6/60
```

**Root Cause**: Service selector mismatch
- Service selector had Helm labels: `app.kubernetes.io/instance: flts`
- Pod labels only had: `app: locust, role: master`
- Result: Service endpoints were empty (`<none>`)

**Resolution**: 
```bash
kubectl delete svc locust-master
kubectl apply -f locust-master.yaml
```
Recreated service with correct selector matching pod labels.

### Issue 2: Initial Deployment Using Environment Variables
**Symptom**: Locust processes starting but not in master/worker mode

**Root Cause**: locustio/locust Docker image doesn't properly read `LOCUST_MODE` environment variable

**Resolution**: Updated deployments to use explicit command-line arguments:
```yaml
command: ["locust"]
args:
  - "--master"  # or "--worker"
  - "--master-bind-host=0.0.0.0"
  - "--locustfile=/home/locust/locustfile.py"
```

Also added `LOCUST_LOCUSTFILE` environment variable as fallback.

## Resource Utilization

### Master Pod
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

### Worker Pods (each)
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

**Total Cluster Resources**:
- Memory: 512Mi + (4 × 256Mi) = 1.5Gi requested, 3Gi limit
- CPU: 500m + (4 × 250m) = 1.5 cores requested, 3 cores limit

## Next Steps

### 1. Deploy Inference Service (Required)
```bash
# Deploy inference deployment and service
kubectl apply -f ../inference-deployment.yaml
kubectl wait --for=condition=ready pod -l app=inference --timeout=120s
```

### 2. Run Full Validation Test
```bash
# Start 50-user test for 60 seconds
curl -X POST http://localhost:30089/swarm \
  -d "user_count=50&spawn_rate=5&host=http://inference:8000"

# Monitor for 60 seconds
watch -n 2 'curl -s http://localhost:30089/stats/requests | jq ".stats[] | select(.name==\"Aggregated\")"'

# Stop test
curl http://localhost:30089/stop
```

### 3. Deploy Driver Job (Optional)
For automated test matrix execution:
```bash
# Build driver image
docker build -t locust-driver:latest -f Dockerfile.driver .

# Apply RBAC
kubectl apply -f locust-driver-job.yaml  # ServiceAccount, Role, RoleBinding

# Run test matrix
kubectl apply -f locust-driver-job-simple.yaml
kubectl logs -f job/locust-load-test-matrix
```

### 4. Enable HPA (Optional)
For automatic inference scaling based on load:
```bash
kubectl autoscale deployment inference --cpu-percent=70 --min=2 --max=8
```

## Validation Checklist

- [x] ConfigMap with locustfile.py deployed
- [x] Locust master deployment running (1 replica)
- [x] Locust worker deployment running (4 replicas)
- [x] Service exposing master on NodePort 30089
- [x] Service endpoints populated correctly
- [x] All 4 workers connected to master
- [x] Web UI accessible at http://localhost:30089
- [x] API endpoint /swarm (start test) working
- [x] API endpoint /stats/requests working
- [x] API endpoint /stop working
- [x] Smoke test executed successfully (system level)
- [ ] Inference service deployed (prerequisite for load tests)
- [ ] End-to-end load test with inference (pending inference deployment)
- [ ] Automated driver job tested (optional)

## Conclusion

The Locust distributed load testing infrastructure is **fully operational** and ready for use. The master and 4 workers are connected, and the system can successfully initiate and execute load tests. 

The high failure rate in the smoke test is expected since the target `inference` service is not yet deployed. Once the inference service is deployed, the system will be able to execute full end-to-end load tests.

### System Status: ✅ READY FOR LOAD TESTING
**Prerequisites**: Deploy inference service to enable full validation

---

**Deployed By**: Automated Deployment  
**Validated By**: System Health Checks + Smoke Test  
**Documentation**: See `.k8s/K8S_LOCUST_DEPLOYMENT_GUIDE.md` for complete usage guide
