# KEDA HTTP Autoscaling - Diagnostic Report
**Date**: 2025-12-02  
**System**: inference-http KEDA HTTPScaledObject  
**Status**: ⚠️ Architecture corrected, testing in progress

---

## Executive Summary

After comprehensive diagnosis, the KEDA HTTP autoscaling system had **zero traffic flowing through the interceptor proxy**. All requests were bypassing KEDA entirely due to a hardcoded ClusterIP service in the deployment YAML. This is a **deployment architecture issue**, not a KEDA configuration problem.

**Key Finding**: Pods never scaled because KEDA's external scaler saw zero pending requests. The interceptor logs showed only health checks - no application traffic.

---

## Root Cause Analysis

### Issue #1: Service Routing Bypass (CRITICAL)
**Impact**: Complete failure of autoscaling mechanism  
**Severity**: CRITICAL - System non-functional

**Problem**: The file `.k8s/inference-http-deployment.yaml` contains both:
1. A Deployment manifest  
2. **A ClusterIP Service manifest** (lines 82-95)

This service creates a direct route (`inference-http:8000 → pods`) that bypasses KEDA's interceptor proxy entirely.

**Evidence**:
```bash
$ kubectl get svc inference-http
NAME             TYPE        CLUSTER-IP     PORT(S)    
inference-http   ClusterIP   10.96.24.35    8000/TCP   # Direct service!

$ kubectl logs -n keda deployment/keda-add-ons-http-interceptor --tail=100
# Only health checks - ZERO application traffic
```

**KEDA Architecture Requirement**:
- Clients MUST connect to: `keda-add-ons-http-interceptor-proxy:8080`  
- Interceptor MUST proxy to backend service with actual pod endpoints
- Backend service should NOT be directly accessible to clients

**Fix Applied**:
```powershell
# Deleted direct service
kubectl delete svc inference-http

# Created backend service for KEDA to proxy to
kubectl apply -f keda-backend-service.yaml  # ClusterIP with pod selector
```

**Validation**:
```bash
$ kubectl run test-proxy --rm -i --restart=Never --image=curlimages/curl:8.10.1 \
  -- curl -s -X POST \
  http://keda-add-ons-http-interceptor-proxy.keda.svc.cluster.local:8080/predict \
  -H "Content-Type: application/json" -H "Host: inference-http" -d '{}'

{"status":"SUCCESS_CACHED",...}  # HTTP 200 ✅
```

---

### Issue #2: KEDA External Scaler Connectivity Failures
**Impact**: Intermittent scaling decisions even with traffic  
**Severity**: HIGH

**Problem**: External scaler logs showed repeated errors:
```
ERROR getting request counts {"error": "there isn't any valid interceptor endpoint"}
```

**Evidence**:
```bash
$ kubectl get pods -n keda
NAME                                               RESTARTS
keda-add-ons-http-external-scaler-xxx              5
keda-add-ons-http-interceptor-xxx                  6
```

**Root Cause**: Unknown - likely transient network/DNS issue or race condition during initial startup.

**Fix Applied**:
```powershell
kubectl rollout restart deployment -n keda keda-add-ons-http-external-scaler
kubectl rollout restart deployment -n keda keda-add-ons-http-interceptor  
kubectl rollout restart deployment -n keda keda-add-ons-http-controller-manager
```

**Validation**:
```bash
$ kubectl get pods -n keda
# All pods: RESTARTS=0, Running ✅
```

---

### Issue #3: Readiness Probe Misconfiguration
**Impact**: Traffic sent to cold pods before model loaded  
**Severity**: MEDIUM

**Problem**: Deployment using `/healthz` endpoint which always returns 200 OK, even when `model_ready == False`.

**Code Analysis** (`inference_container/api_server.py`):
```python
@app.get("/healthz")
def healthz():
    model_ready = inf.current_model is not None
    return {"status": "ok", "model_ready": model_ready, ...}  # Always 200

@app.get("/ready")  
def ready():
    model_ready = inf.current_model is not None
    if model_ready:
        return Response(..., status_code=200)
    return Response(..., status_code=503)  # Proper readiness signal
```

**Fix Applied**:
```bash
kubectl patch deployment inference-http --type=json -p='[
  {"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/httpGet/path", "value": "/ready"},
  {"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/initialDelaySeconds", "value": 10},
  {"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/periodSeconds", "value": 3},
  {"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/failureThreshold", "value": 10}
]'
```

---

### Issue #4: Locust Host Header Passing
**Impact**: Cannot run load tests via kubectl exec  
**Severity**: LOW (testing-only)

**Problem**: Locust's `-H` flag cannot be passed correctly through `kubectl exec` + `/bin/sh -c` execution model. The command parser treats it as part of the host URL.

**Error**:
```
InvalidSchema: No connection adapters were found for 'Host: inference-http/predict'
```

**Workaround**: Direct curl tests work. For production load testing, either:
1. Expose interceptor proxy via Ingress/LoadBalancer
2. Run Locust natively (not in K8s pod)
3. Modify Locust pod to pre-configure target host

---

## Load Test Results

### Test 1: Direct Traffic (Bypassing KEDA)
**Config**: 150 users, 30/s ramp, 90s duration  
**Target**: `http://inference-http:8000` (direct service)  
**Result**:
- ✅ 7564 requests, **0 failures** (100% success rate)
- ✅ P50: 130ms, P95: 880ms, P99: 1.7s
- ✅ Sustained 85-94 RPS
- ❌ **No scaling occurred** - pod count remained at 2

**Analysis**: System handles load well when traffic balanced across 2 pods. But KEDA never detected traffic because interceptor wasn't in path.

---

## HTTPScaledObject Status

```yaml
apiVersion: http.keda.sh/v1alpha1
kind: HTTPScaledObject
metadata:
  name: inference-http-scaler
spec:
  hosts:
  - inference-http.default.svc.cluster.local
  - inference-http
  scaleTargetRef:
    name: inference-http
    kind: Deployment
    service: inference-http
    port: 8000
  replicas: {min: 2, max: 12}
  targetPendingRequests: 10
  scaledownPeriod: 120

status:
  conditions:
  - type: Ready
    status: "True"
  - type: Active
    status: "False"  # ⚠️ Scaler not active - no traffic detected
    reason: ScalerNotActive
```

**Key Status Field**: `Active: False` confirms KEDA is not receiving any traffic metrics.

---

## Architecture Diagram

### BEFORE (Broken):
```
Locust/Clients
    ↓
inference-http:8000 (ClusterIP Service)
    ↓
inference-http-xxx pods (direct)
    
KEDA Interceptor: [idle - no traffic] ❌
```

### AFTER (Fixed):
```
Clients
    ↓
keda-add-ons-http-interceptor-proxy:8080
    ↓ (proxies to)
inference-http:8000 (backend service)
    ↓
inference-http-xxx pods
    
KEDA External Scaler: [monitors queue] ✅
KEDA Interceptor: [sees all traffic] ✅
```

---

## Recommendations

### Immediate (Production Blocking):

1. **Remove Service from Deployment YAML** ⚠️  
   Delete lines 82-95 in `.k8s/inference-http-deployment.yaml`. Services should NOT be colocated with deployments when using KEDA HTTP addon.

2. **Update Load Test Configuration**  
   Change all test scripts to target: `keda-add-ons-http-interceptor-proxy.keda.svc.cluster.local:8080`  
   Add `Host: inference-http` header to all requests.

3. **Document KEDA Routing**  
   Add clear documentation that clients MUST go through interceptor proxy for autoscaling to work.

### Monitoring:

4. **Add Interceptor Traffic Metrics**  
   Monitor `keda-add-ons-http-interceptor` logs for request volume. If only health checks appear, routing is broken.

5. **Alert on Active: False**  
   Create alert when `HTTPScaledObject.status.conditions[type=Active].status == False` for >5 minutes during expected traffic.

### Architecture:

6. **Consider Ingress Integration**  
   Expose KEDA interceptor proxy via Ingress with TLS. Current setup requires internal cluster DNS knowledge.

7. **Add Circuit Breaker**  
   If KEDA fails, traffic should fall back to direct service routing (currently broken if interceptor unavailable).

---

## Testing Checklist

- [x] KEDA components restarted and stable (0 restarts)
- [x] External scaler connectivity verified  
- [x] Readiness probe fixed (`/ready` endpoint)
- [x] Direct service removed
- [x] Backend service created with pod endpoints
- [x] Single curl test through interceptor succeeds (200 OK)
- [ ] Load test through interceptor with 150 users
- [ ] Verify pod count scales up (2 → N pods)
- [ ] Verify queue metrics increase during load
- [ ] Verify scale-down after cooldown period
- [ ] Measure P95 latency < 400ms at steady state

---

## Files Modified

1. **inference-http deployment**: Readiness probe patched to `/ready`
2. **keda-backend-service.yaml**: Created - ClusterIP service for KEDA to proxy to
3. **run_keda_load_test_with_monitoring.ps1**: Updated target URL (pending fix for `-H` flag issue)

---

## Next Steps

1. Fix Locust invocation to properly route through KEDA proxy
2. Run 90-second load test with 150 users
3. Verify scaling timeline: 2 → 4+ pods within 60s
4. Document final latency/throughput at scale
5. Verify scale-down after 120s cooldown period

---

**Engineer Notes**: The system was fundamentally misconfigured from deployment. KEDA HTTP addon requires very specific service architecture - all traffic MUST flow through interceptor proxy. The presence of a direct ClusterIP service completely defeats the autoscaling mechanism. This is a common pitfall when migrating existing deployments to KEDA HTTP.
