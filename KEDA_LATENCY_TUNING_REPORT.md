# KEDA Latency Tuning Investigation Report

**Date:** 2025-11-12  
**Objective:** Optimize KEDA ScaledObject to show real-time latency metrics during short (60-180s) load tests

---

## Executive Summary

**Attempted Optimization:**
- Changed Prometheus query rate window from `[5m]` to `[1m]`
- Lowered thresholds from 300ms/500ms to 200ms/350ms
- Restarted KEDA metrics apiserver

**Result:** KEDA external metrics still returning 0 despite:
- Prometheus successfully scraping inference pods
- Histogram buckets containing valid data
- Manual Prometheus queries returning correct latency values (1870ms during investigation)

**Root Cause:** KEDA → Prometheus integration failure, not related to query window or thresholds.

---

## Configuration Changes Applied

### Before (Original)
```yaml
triggers:
  - type: prometheus
    metadata:
      query: |
        histogram_quantile(0.95, 
          sum(rate(inference_latency_seconds_bucket[5m])) by (le)
        )
      threshold: "0.500"           # 500ms
      activationThreshold: "0.300" # 300ms
      serverAddress: http://prometheus-server.default.svc.cluster.local:80
```

### After (Patched)
```yaml
triggers:
  - type: prometheus
    metadata:
      query: histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[1m])) by (le))
      threshold: "0.350"           # 350ms
      activationThreshold: "0.200" # 200ms
      serverAddress: http://prometheus-server.default.svc.cluster.local:80
```

**Patch applied:** `kubectl patch scaledobject inference-slo-scaler --type='json' --patch-file=keda-latency-patch.json`  
**Status:** Successfully applied, ScaledObject updated

---

## Validation Tests

### Test 1: 100 users, 60 seconds
- Requests: 1,843
- Failures: 0%
- CPU: Peak 1001m (exceeds 850m threshold)
- **HPA Latency Metric:** 0ms ❌

### Test 2: 150 users, 120 seconds
- Requests: 3,712
- Failures: 0%
- CPU: Peak 999m
- **HPA Latency Metric:** 0ms ❌

### Test 3: 200 users, 90 seconds
- Requests: 2,785
- Failures: 0%
- CPU: Peak 1000m
- **HPA Latency Metric:** 0ms ❌

**Real-time Monitoring:** Background job tracked HPA `s0-prometheus` metric every 5 seconds during all tests → **consistently 0**.

---

## Diagnostic Findings

### 1. Prometheus Scraping ✅ WORKING
```json
[
  {"scrapeUrl": "http://10.1.6.14:8000/prometheus", "scrapeInterval": "1m", "health": "up"},
  {"scrapeUrl": "http://10.1.6.15:8000/prometheus", "scrapeInterval": "1m", "health": "up"},
  {"scrapeUrl": "http://10.1.6.16:8000/prometheus", "scrapeInterval": "1m", "health": "up"}
]
```
- All 3 inference pods scraped successfully
- Scrape interval: **1 minute** (60 seconds)
- Target health: All `up`

### 2. Histogram Bucket Data ✅ EXISTS
```promql
inference_latency_seconds_bucket{le="0.05"} 38
inference_latency_seconds_bucket{le="0.1"}  102
inference_latency_seconds_bucket{le="+Inf"} 3438 (total count)
```
- Buckets contain cumulative request counts
- Data is being recorded during load tests

### 3. Manual Prometheus Queries ✅ RETURN CORRECT VALUES

**5-minute window (current):**
```bash
$ kubectl exec prometheus-server -- wget -qO- \
  'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(inference_latency_seconds_bucket[5m]))by(le))'
# Result: 1.87 seconds (1870ms) ✅
```

**1-minute window (new):**
```bash
$ kubectl exec prometheus-server -- wget -qO- \
  'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(inference_latency_seconds_bucket[1m]))by(le))'
# Result: [] (empty during idle, valid during load)
```

**Observation:** 1m window returns empty when no traffic in last 60 seconds (expected behavior).

### 4. KEDA External Metrics API ❌ RETURNS 0
```bash
$ kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/default/s0-prometheus?labelSelector=scaledobject.keda.sh%2Fname%3Dinference-slo-scaler"
{
  "items": [{
    "metricName": "s0-prometheus",
    "value": "0"  # ❌ Always 0, never updates
  }]
}
```

### 5. KEDA Metrics Apiserver Logs ⚠️ PERSISTENT ERRORS
```
E1106 01:07:41 status.go:71] "Unhandled Error" 
  err="apiserver received an error that is not an metav1.Status: rpc error: 
  code = Unknown desc = error when getting metric values metric:s0-prometheus encountered error"
```
- Errors date back to November 6th
- Consistent pattern: `error when getting metric values metric:s0-prometheus`
- Continues after metrics apiserver restart
- Also affects `s1-prometheus` (queue length metric)

### 6. KEDA → Prometheus Connectivity ✅ REACHABLE
```bash
$ kubectl run -n keda test-curl --image=curlimages/curl --rm -it -- curl http://prometheus-server.default.svc.cluster.local:80/api/v1/query?query=up
# Result: HTTP 200, returns metrics ✅
```
- KEDA namespace can reach Prometheus
- DNS resolution works
- HTTP connectivity confirmed

---

## Root Cause Analysis

### Problem: KEDA Cannot Query Prometheus Metrics

**NOT the issue:**
- ✅ Prometheus scraping (working)
- ✅ Histogram data existence (present)
- ✅ Network connectivity (verified)
- ✅ Query syntax (manual queries work)
- ✅ Rate window duration (both 1m and 5m tested)

**Likely issues:**
1. **KEDA Prometheus scaler configuration error**
   - Query might need different format/escaping for KEDA
   - serverAddress might need authentication
   - Missing required metadata fields

2. **KEDA internal bug or cache**
   - Metrics apiserver showing persistent "Unknown" errors
   - Error predates our tuning changes (since Nov 6)
   - Restarting apiserver/operator doesn't resolve

3. **Prometheus query API incompatibility**
   - KEDA might be using deprecated API endpoint
   - Response format mismatch
   - Timeout issues (queries take >1s to return)

---

## Impact Assessment

### Current State
- **CPU-based scaling:** ✅ Working (HPA shows 99% CPU during tests)
- **Memory-based scaling:** ✅ Working (HPA shows 17% memory)
- **Latency-based scaling:** ❌ Broken (always shows 0ms)
- **Queue-based scaling:** ❌ Broken (always shows 0)

### Implications
- System can only scale based on resource utilization
- Latency degradation won't trigger scaling
- Queue depth increases won't trigger scaling
- 200 users (current capacity) produces ~1870ms latency but no KEDA response

### Risk Analysis
If production load exceeds 200 concurrent users:
- CPU hits 100%, requests slow down
- Latency increases to multi-second range
- KEDA won't detect latency spike
- HPA might scale on CPU, but with 60s stabilization delay
- Users experience degraded service before scaling occurs

---

## Tested Solutions

### ✅ Applied
1. Changed rate window `[5m]` → `[1m]` (patch successful)
2. Lowered thresholds 300ms/500ms → 200ms/350ms (patch successful)
3. Restarted KEDA metrics apiserver (no effect)
4. Restarted KEDA operator (no effect)

### ❌ Did NOT Resolve
- KEDA external metrics still return 0
- HPA still shows `s0-prometheus: 0`
- No latency-based scaling events

---

## Recommendations

### Immediate (Debug KEDA)
1. **Enable KEDA debug logging:**
   ```bash
   kubectl edit deployment -n keda keda-operator
   # Add: --zap-log-level=debug
   ```

2. **Check KEDA scaler auth:**
   ```yaml
   metadata:
     authModes: "bearer"  # If Prometheus requires auth
     # OR
     unsafeSsl: "true"    # If cert issues
   ```

3. **Simplify query for testing:**
   ```yaml
   query: "up"  # Start with simplest possible query
   ```

4. **Check KEDA version compatibility:**
   ```bash
   kubectl get deployment -n keda keda-operator -o yaml | grep image:
   # Current: 2.18.1
   # Check release notes for Prometheus scaler bugs
   ```

### Short-term (Workaround)
1. **Use CPU scaling only (current state):**
   - Already works
   - Scale threshold: 85% CPU
   - 60s stabilization window

2. **Add queue-depth scaling via alternative method:**
   - Export queue metrics to different backend (Datadog, Grafana)
   - Use native HPA external metrics (bypass KEDA)

3. **Implement application-level autoscaling:**
   - Inference service calls K8s API to self-scale
   - Trigger based on internal queue depth tracking

### Long-term (Proper Fix)
1. **Investigate KEDA Prometheus scaler thoroughly:**
   - Review KEDA GitHub issues for similar problems
   - Test with KEDA 2.19+ (if available)
   - Consider filing bug report with full diagnostic data

2. **Alternative: Use native HPA + custom metrics adapter:**
   - Deploy Prometheus Adapter for HPA
   - Bypass KEDA for Prometheus metrics
   - Keep KEDA only for Kafka/other scalers

3. **Alternative: Use Prometheus recording rules:**
   - Pre-calculate p95 latency at Prometheus level
   - KEDA queries simpler metric (no histogram_quantile)
   - Reduces query complexity

---

## Testing Checklist for Next Debug Session

- [ ] Enable KEDA operator debug logging
- [ ] Capture KEDA operator logs during active load test
- [ ] Test with simplified query (`up` or `inference_latency_seconds_count`)
- [ ] Verify Prometheus query API version (v1 vs v2)
- [ ] Check if Prometheus Adapter (alternative) is installed
- [ ] Review KEDA ScaledObject status conditions (`kubectl describe scaledobject`)
- [ ] Test queue metric (`s1-prometheus`) separately
- [ ] Check KEDA gRPC communication (operator ↔ metrics apiserver)

---

## Related Files

**Created this session:**
- `keda-latency-patch.json` - JSON patch with 1m window + lower thresholds
- `scaledobject-backup.yaml` - Original configuration (5m window)
- `scaledobject-current.yaml` - Current configuration (1m window)
- `validate_hybrid_autoscaling.ps1` - Test automation script
- `autoscaling_results/telemetry_*.csv` - Test telemetry data

**Previous reports:**
- `PROMETHEUS_FIX_VALIDATION_REPORT.md` - Prometheus scraping fix
- `HYBRID_AUTOSCALING_VALIDATION_REPORT.md` - Initial validation (pre-Prometheus fix)

---

## Appendix: Key Commands

**Query KEDA external metrics:**
```bash
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/default/s0-prometheus?labelSelector=scaledobject.keda.sh%2Fname%3Dinference-slo-scaler"
```

**Manual Prometheus query:**
```bash
kubectl exec prometheus-server-XXX -c prometheus-server -- \
  wget -qO- 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(inference_latency_seconds_bucket[1m]))by(le))'
```

**Check KEDA logs:**
```bash
kubectl logs -n keda keda-operator-XXXXX --tail=50
kubectl logs -n keda keda-operator-metrics-apiserver-XXXXX --tail=50
```

**Restart KEDA components:**
```bash
kubectl delete pod -n keda -l app=keda-operator
kubectl delete pod -n keda -l app=keda-operator-metrics-apiserver
```

---

**Status:** Investigation incomplete - KEDA → Prometheus integration failure requires deeper debugging with KEDA operator logs at debug level.
