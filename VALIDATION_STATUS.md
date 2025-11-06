# Current Validation Status - HPA Testing

## What Was Attempted

1. **SLO-Based Autoscaling Implementation** ✅ Design Complete
   - Added HTTP-level Prometheus metrics to `api_server.py`
   - Created HPA v2 manifest with custom metrics (P95 latency, in-flight requests, CPU)
   - Created KEDA alternative scaler
   - Created comprehensive deployment guide
   - Created before/after comparison document

2. **Deployment Challenges** ⚠️ Blocked
   - Attempted to apply SLO deployment
   - Encountered syntax errors in instrumentation code
   - Reverted to original code
   - Need Prometheus infrastructure before full SLO deployment

## Current State

**Infrastructure:**
- ✅ Kubernetes cluster running
- ✅ Inference service deployed (2 replicas)
- ✅ HPA configured (CPU-based, min=2, max=20, target=70%)
- ✅ Locust load testing service available
- ❌ Prometheus not installed
- ❌ Prometheus Adapter not installed
- ❌ KEDA not installed

**Testing Script:**
- ✅ `k8s_auto_hpa_tests.ps1` updated for HPA-driven testing
- ✅ HPA testing guide documented (`HPA_TESTING_GUIDE.md`)
- ⚠️ Last test run failed (will validate after rebuild)

## Recommended Next Steps

### Option A: Validate Current HPA (CPU-Only) - RECOMMENDED FOR NOW

This validates that the HPA-driven testing framework works before adding complexity.

**Steps:**
1. ✅ Rebuild inference container (in progress)
2. ⏳ Restart deployment with clean image
3. ⏳ Verify pods are healthy and ready
4. ⏳ Run simple validation test:
   ```powershell
   kubectl exec deployment/locust-master -- sh -c "cd /home/locust && locust -f locustfile.py --headless --host=http://inference:8000 -u 50 -r 5 -t 60s"
   ```
5. ⏳ Watch HPA scaling:
   ```powershell
   kubectl get hpa inference -w
   ```
6. ⏳ Run HPA test matrix:
   ```powershell
   .\scripts\k8s_auto_hpa_tests.ps1 `
     -UserCounts @(50,100,200) `
     -WorkerCounts @(4) `
     -HPAMinReplicas 2 `
     -HPAMaxReplicas 15 `
     -HPATargetCPU 70 `
     -TestDuration 120 `
     -OutputDir "reports\hpa_cpu_baseline"
   ```
7. ⏳ Collect baseline metrics:
   - P50/P95/P99 latency at different load levels
   - Replica count vs load
   - Scale-up and scale-down timing
   - CPU utilization patterns

**Expected Outcomes:**
- HPA scales from 2 → 4-6 replicas under 100-200 user load
- CPU stays below 80% after scaling
- P95 latency ~300-500ms (baseline, no SLO enforcement)
- Slow scale-down after load drops (5min default)

### Option B: Install Prometheus & Deploy SLO - DEFERRED

This is the full implementation but requires infrastructure setup.

**Prerequisites:**
1. Install Prometheus Operator:
   ```bash
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace
   ```

2. Install Prometheus Adapter:
   ```bash
   helm install prometheus-adapter prometheus-community/prometheus-adapter --namespace monitoring
   ```

3. Configure custom metrics (see `SLO_DEPLOYMENT_GUIDE.md`)

4. Fix syntax errors in `api_server.py` metrics instrumentation

5. Rebuild and deploy with `.kubernetes/inference-deployment-slo.yaml`

6. Apply `.kubernetes/inference-slo-hpa.yaml`

**Time Estimate:** 2-3 hours for full setup and validation

## Current Test Command Status

**Last Attempted:**
```powershell
.\scripts\k8s_auto_hpa_tests.ps1 `
  -UserCounts @(50,100,200,400) `
  -WorkerCounts @(4,8) `
  -HPAMinReplicas 2 `
  -HPAMaxReplicas 15 `
  -HPATargetCPU 70 `
  -TestDuration 120
```

**Result:** Exit code 1 (likely due to inference pods not ready)

**Next Attempt:** Wait for rebuild to complete, verify pods healthy, then retry with simpler test:
```powershell
.\scripts\k8s_auto_hpa_tests.ps1 `
  -UserCounts @(50) `
  -WorkerCounts @(4) `
  -HPAMinReplicas 2 `
  -HPAMaxReplicas 10 `
  -HPATargetCPU 70 `
  -TestDuration 60 `
  -OutputDir "reports\hpa_validation"
```

## Files Created This Session

### SLO Implementation (Design Complete, Deployment Pending)
1. `.kubernetes/inference-deployment-slo.yaml` - Deployment with resource limits and metrics port
2. `.kubernetes/inference-slo-hpa.yaml` - HPA v2 with custom metrics + ServiceMonitor + PrometheusRule
3. `.kubernetes/inference-keda-scaler.yaml` - KEDA alternative
4. `.kubernetes/SLO_DEPLOYMENT_GUIDE.md` - 600+ line comprehensive deployment guide
5. `SLO_BEFORE_AFTER.md` - Executive summary and comparison

### Modified Files
1. `inference_container/api_server.py` - HTTP metrics added (reverted due to syntax errors)

## Decision Point

**RECOMMENDATION: Proceed with Option A (CPU-Only HPA Validation)**

**Rationale:**
1. Validates that HPA-driven testing framework works correctly
2. Establishes baseline metrics for comparison
3. No infrastructure dependencies (Prometheus, adapters)
4. Quick feedback loop (30 minutes vs 3 hours)
5. Can proceed to Option B once baseline is established

**Success Criteria for Option A:**
- ✅ Inference pods healthy and responding
- ✅ Locust can generate load successfully
- ✅ HPA scales up under load (CPU > 70%)
- ✅ HPA scales down after load drops
- ✅ Test script collects metrics correctly
- ✅ CSV output includes replica counts, latencies, CPU%

## Validation Commands

Once rebuild completes:

```powershell
# 1. Check pod health
kubectl get pods -l app=inference

# 2. Test inference endpoint
kubectl exec deployment/inference -- python -c "import requests; r = requests.post('http://localhost:8000/predict', json={'inference_length': 1}); print(f'Status: {r.status_code}')"

# 3. Check HPA status
kubectl get hpa inference
kubectl describe hpa inference

# 4. Quick load test
kubectl exec deployment/locust-master -- sh -c "cd /home/locust && locust -f locustfile.py --headless --host=http://inference:8000 -u 30 -r 5 -t 60s" 2>&1 | Select-String "Aggregated"

# 5. Watch scaling in real-time
kubectl get hpa inference -w

# 6. Run automated test suite
.\scripts\k8s_auto_hpa_tests.ps1 -UserCounts @(50,100) -WorkerCounts @(4) -TestDuration 120 -OutputDir "reports\hpa_baseline"
```

## Next Session Plan

**If Option A Succeeds:**
1. Analyze baseline HPA performance
2. Install Prometheus stack
3. Fix api_server.py metrics instrumentation
4. Deploy SLO-based HPA
5. Run comparison tests (CPU-only vs SLO-driven)
6. Document improvements

**If Option A Fails:**
- Debug inference service health issues
- Check Kafka/MLflow dependencies
- Verify model promotion artifacts exist
- Test manual predictions before load testing

## Time Estimates

- **Option A (CPU-HPA Validation):** 30-45 minutes
  - Rebuild complete: 5 min
  - Pod rollout: 2 min
  - Health checks: 5 min
  - Test execution: 15-20 min
  - Analysis: 10 min

- **Option B (Full SLO):** 2-3 hours
  - Prometheus install: 15 min
  - Adapter setup: 30 min
  - Code fixes: 30 min
  - Deployment: 15 min
  - Validation: 45-60 min
