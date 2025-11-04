# FLTS Inference Service - Load Test Results & Performance Analysis

**Test Date:** November 4, 2025 11:26 AM  
**Environment:** Kubernetes (Docker Desktop)  
**Inference Replicas:** 2 pods  
**Endpoint:** http://localhost/predict (LoadBalancer)  
**Model:** LSTM (run_id: 4a4e0e5182934d0780520ca6f610b9d2)

---

## Executive Summary

✅ **System handled 320 total requests across 3 test phases with 100% success rate**

**Key Findings:**
- ✅ **Zero Errors:** All 320 requests completed successfully (100% success rate)
- ⚠️ **Latency Degradation:** Significant increase (473%) when scaling from 10 to 50 concurrent users
- ✅ **Throughput Stable:** Maintained 16-21 req/s across all test phases
- ✅ **No Crashes:** Both inference pods remained stable throughout testing
- ⚠️ **Performance Bottleneck:** p95 latency exceeded 5 seconds at 100 concurrent users

**Recommendation:** System performs well up to 10 concurrent users. Beyond that, horizontal scaling (3-5+ replicas) or increased concurrency limits needed.

---

## Test Configuration

### Progressive Load Phases

| Phase | Concurrent Users | Requests/User | Total Requests | Duration |
|-------|------------------|---------------|----------------|----------|
| 1 | 10 | 2 | 20 | 1.22s |
| 2 | 50 | 2 | 100 | 5.96s |
| 3 | 100 | 2 | 200 | 9.42s |
| **Total** | - | - | **320** | **16.60s** |

### Infrastructure

**Inference Pods:**
- Replicas: 2
- Pod Names: 
  - `inference-8645cf58c7-44dnr`
  - `inference-8645cf58c7-f9wnm`
- Resource Limits:
  - CPU: 1 core (requests: 500m)
  - Memory: 2Gi (requests: 1Gi)

**Service Configuration:**
- Type: LoadBalancer
- Internal: ClusterIP 10.107.123.158:8000
- External: localhost:80 (NodePort 30716)
- Concurrency Limit: 16 workers per pod
- Mode: Synchronous prediction

---

## Performance Results

### Phase 1: 10 Concurrent Users (Baseline)

**Request Statistics:**
- Total Requests: 20
- Success Rate: 100%
- Duration: 1.22 seconds
- Throughput: **16.43 req/s**

**Latency Distribution:**
```
Min:    368 ms
Mean:   519 ms
Median: 520 ms
p95:    686 ms
p99:    686 ms
Max:    686 ms
```

**Analysis:** ✅ **EXCELLENT PERFORMANCE**
- Sub-second latencies for all requests
- Consistent response times (min-max range: 318ms)
- System operating well within capacity

---

### Phase 2: 50 Concurrent Users

**Request Statistics:**
- Total Requests: 100
- Success Rate: 100%
- Duration: 5.96 seconds
- Throughput: **16.79 req/s**

**Latency Distribution:**
```
Min:    1,030 ms
Mean:   2,795 ms
Median: 2,591 ms
p95:    3,930 ms
p99:    5,821 ms
Max:    5,821 ms
```

**Analysis:** ⚠️ **SIGNIFICANT DEGRADATION**
- p95 latency increased **473%** (686ms → 3,930ms)
- Mean latency increased **439%** (519ms → 2,795ms)
- Max latency increased **749%** (686ms → 5,821ms)
- Queue wait times observed: 358-489ms per request
- Success rate remains 100% (no errors)

**Root Cause:** Concurrency limit (16 workers per pod × 2 pods = 32 total) insufficient for 50 simultaneous connections. Requests queuing behind active workers.

---

### Phase 3: 100 Concurrent Users

**Request Statistics:**
- Total Requests: 200
- Success Rate: 100%
- Duration: 9.42 seconds
- Throughput: **21.23 req/s** (highest)

**Latency Distribution:**
```
Min:    1,050 ms
Mean:   4,134 ms
Median: 4,217 ms
p95:    5,463 ms
p99:    5,830 ms
Max:    6,361 ms
```

**Analysis:** ⚠️ **CONTINUED DEGRADATION (but slowing)**
- p95 latency increased **39%** from Phase 2 (3,930ms → 5,463ms)
- Mean latency increased **48%** from Phase 2 (2,795ms → 4,134ms)
- Throughput actually improved slightly (16.79 → 21.23 req/s)
- Event loop lag peaked at 1,707ms during test
- Max execution time observed: 3,347ms

**Observation:** System saturated at ~32 concurrent connections. Additional load doesn't drastically worsen performance but maintains high latency floor.

---

## Comparative Analysis

### Latency Progression

| Metric | 10 Users | 50 Users | 100 Users | Change (10→50) | Change (50→100) |
|--------|----------|----------|-----------|----------------|-----------------|
| **p50 (median)** | 520ms | 2,591ms | 4,217ms | +398% | +63% |
| **p95** | 686ms | 3,930ms | 5,463ms | +473% | +39% |
| **p99** | 686ms | 5,821ms | 5,830ms | +749% | +0.2% |
| **Mean** | 519ms | 2,795ms | 4,134ms | +439% | +48% |

### Throughput Stability

| Phase | Concurrent Users | Throughput (req/s) | Efficiency* |
|-------|------------------|-------------------|-------------|
| 1 | 10 | 16.43 | 1.64 req/s/user |
| 2 | 50 | 16.79 | 0.34 req/s/user |
| 3 | 100 | 21.23 | 0.21 req/s/user |

*Efficiency = Throughput / Concurrent Users (higher is better)

**Key Insight:** Throughput remains relatively stable (16-21 req/s) but efficiency drops dramatically. System is processing requests as fast as possible, but queue backlogs cause latency increase.

---

## Resource Utilization Analysis

### Inference Service Metrics (Post-Test)

**Queue & Concurrency:**
- Current queue length: 0 (cleared after test)
- Total workers: 16 per pod
- Active workers (at observation): 0
- Total completed requests: 166 (on observed pod)
- Error count: 0

**Latency Metrics (Observed Pod):**
- Last inference duration: 605ms
- Max inference duration: 3,347ms
- Average inference duration: 985ms
- Last queue wait: 489ms
- Max queue wait: 1,174ms
- Average queue wait: 165ms

**Data Preparation:**
- Last prep time: 31ms
- Max prep time: 202ms
- Average prep time: 40ms

**Event Loop Performance:**
- Last lag: 2.21ms
- Max lag: 1,707ms (peak during high load)
- Average lag: 4.70ms

### Resource Limits vs Usage

**Configured Resources (per pod):**
- CPU Limit: 1 core
- CPU Request: 500m (0.5 core)
- Memory Limit: 2Gi
- Memory Request: 1Gi

**Observations:**
- No OOMKilled events observed
- Pods remained Running throughout test
- No restarts or crashes
- CPU likely at or near limit during peak load (inferred from event loop lag)

**Note:** Kubernetes metrics-server not available in Docker Desktop - precise CPU/memory usage percentages unavailable.

---

## Performance Bottleneck Identification

### Primary Bottleneck: Concurrency Limit

**Evidence:**
1. **Stable Throughput:** 16-21 req/s across all phases indicates processing capacity ceiling
2. **Queue Wait Times:** 165ms average, 1,174ms max shows requests waiting for workers
3. **Event Loop Lag:** 1,707ms max indicates CPU saturation during peak
4. **Worker Saturation:** 3-4 active workers observed during load (16 configured)

**Calculation:**
- Per-request processing time: ~1,000ms (985ms avg observed)
- Concurrency limit: 16 workers/pod × 2 pods = 32 concurrent
- Theoretical max throughput: 32 workers / 1s = 32 req/s
- Observed throughput: 16-21 req/s (50-66% of theoretical max)

**Why not reaching theoretical max?**
- Network latency (~50-100ms round trip)
- Data preparation overhead (~40ms)
- Queue management overhead
- Load balancer distribution not perfectly even

### Secondary Bottleneck: CPU

**Evidence:**
1. **Event Loop Lag:** 1,707ms peak suggests CPU starvation
2. **Inference Duration:** 985ms average with 3,347ms max indicates computation-bound
3. **Resource Limits:** 1 CPU core per pod may be insufficient for LSTM inference

**LSTM Model Characteristics:**
- Sequential computation (cannot parallelize within single prediction)
- Tensor operations (CPU-bound without GPU)
- Input sequence length: 10 timesteps
- 30 input samples with 11 features each

---

## Degradation Threshold Analysis

### Performance Tiers

**Tier 1: Acceptable Performance (≤10 concurrent users)**
- Latency: p95 < 700ms, p99 < 700ms
- Throughput: 16+ req/s
- Success Rate: 100%
- **Status:** ✅ **MEETS SLA**

**Tier 2: Degraded Performance (11-50 concurrent users)**
- Latency: p95 2,000-4,000ms, p99 4,000-6,000ms
- Throughput: 16-21 req/s
- Success Rate: 100%
- **Status:** ⚠️ **FUNCTIONAL BUT SLOW**

**Tier 3: Saturated (50+ concurrent users)**
- Latency: p95 > 5,000ms, p99 > 5,500ms
- Throughput: 21 req/s (ceiling)
- Success Rate: 100%
- **Status:** ⚠️ **SYSTEM AT CAPACITY**

### Critical Thresholds Identified

| Threshold | Value | Consequence |
|-----------|-------|-------------|
| **Latency Warning** | 10→50 users | p95 latency jumps 473% |
| **Latency Critical** | 50+ users | p95 exceeds 5 seconds |
| **Throughput Ceiling** | ~21 req/s | Cannot process faster |
| **Queue Buildup** | 32+ concurrent | Requests wait 100-1,000ms |
| **Event Loop Saturation** | Peak load | 1.7s lag indicates CPU starvation |

**Recommendation:** System should target **<10 concurrent users per 2-pod deployment** to maintain sub-second latency.

---

## Recommendations

### Immediate Actions (Current Environment)

1. **Reduce Expected Concurrency:**
   - Current suitable for: ≤10 concurrent users
   - For interactive workloads requiring <1s latency

2. **Increase Concurrency Limit (if latency tolerance higher):**
   ```yaml
   # values-complete.yaml
   inference:
     env:
       PREDICT_MAX_CONCURRENCY: "32"  # Currently 16
   ```
   - Would reduce queue waits
   - May increase CPU contention

3. **Enable Request Caching:**
   ```yaml
   inference:
     env:
       ENABLE_PREDICT_CACHE: "1"  # Already enabled
   ```
   - Helps with repeated identical requests
   - Not effective for unique time-series predictions

### Production Scaling (Cloud Cluster)

#### Horizontal Scaling

**For 50 concurrent users (p95 < 2s):**
- Increase replicas: 2 → 5
- Expected throughput: ~80 req/s (16 × 5)
- Target: 10 users per pod (50 / 5 = 10)

**For 100 concurrent users (p95 < 2s):**
- Increase replicas: 2 → 10
- Expected throughput: ~160 req/s
- Target: 10 users per pod (100 / 10 = 10)

**Configuration:**
```yaml
# values-complete.yaml
inference:
  replicas: 10
  
# Or use HPA (Horizontal Pod Autoscaler)
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    kind: Deployment
    name: inference
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 30
```

#### Vertical Scaling (Increase Pod Resources)

**Current:**
- CPU: 1 core limit, 500m request
- Memory: 2Gi limit, 1Gi request

**Recommended for Production:**
```yaml
inference:
  resources:
    requests:
      cpu: "2000m"  # 2 cores
      memory: "3Gi"
    limits:
      cpu: "4000m"  # 4 cores
      memory: "6Gi"
```

**Expected Improvement:**
- 2× CPU → ~1.5-1.8× throughput per pod
- Reduced event loop lag
- Lower inference duration (985ms → ~500-700ms)

#### GPU Acceleration (Optimal Solution)

**If available (AWS EKS with GPU nodes, GKE with GPUs):**
```yaml
inference:
  resources:
    limits:
      nvidia.com/gpu: 1
  nodeSelector:
    accelerator: nvidia-tesla-t4
```

**Expected Improvement:**
- LSTM inference: 985ms → 50-100ms (10-20× faster)
- Throughput: 21 req/s → 200+ req/s per pod
- Latency: p95 from 5,000ms → 500ms at 100 concurrent

---

## Load Testing Best Practices

### Recommended Test Plan for Production

**Phase 1: Warm-up**
- 5 users × 60 seconds
- Verify model loaded and responding
- Baseline latency measurement

**Phase 2: Ramp-up**
- Increase 5 users every 60 seconds
- 5 → 10 → 15 → 20 → 25 → 30 users
- Identify latency degradation point

**Phase 3: Sustained Load**
- Target concurrency (e.g., 20 users)
- Duration: 10-30 minutes
- Monitor for memory leaks, gradual degradation

**Phase 4: Spike Test**
- Sudden increase to 2× or 3× sustained load
- Hold for 2-5 minutes
- Verify recovery to baseline

**Phase 5: Soak Test**
- Run at 50-70% capacity
- Duration: 2-24 hours
- Verify stability over time

### Metrics to Collect

**Per-Request Metrics:**
- Latency (p50, p95, p99, max)
- Status codes (200, 4xx, 5xx)
- Request size, response size

**System Metrics:**
- CPU usage per pod (%)
- Memory usage per pod (%)
- Network I/O
- Disk I/O (for MLflow/MinIO operations)

**Application Metrics:**
- Queue length over time
- Active workers
- Event loop lag
- Cache hit rate
- Model inference time
- Data prep time

---

## Comparison to Requirements

### Current Performance vs Production Targets

| Metric | Current (2 pods) | Production Target | Gap |
|--------|------------------|-------------------|-----|
| **Max Concurrent Users** | 10 (for <1s latency) | 100+ | Need 10× scale |
| **Throughput** | 16-21 req/s | 100-200 req/s | Need 5-10× scale |
| **p95 Latency @ 100 users** | 5,463ms | <2,000ms | Need 3× improvement |
| **Success Rate** | 100% ✅ | 99.9%+ ✅ | **MEETS** |
| **Uptime** | 100% (test duration) | 99.9% | **MEETS** |

### Path to Production Targets

**Option 1: Horizontal Scaling Only**
- Deploy 10 replicas (5× current)
- Expected throughput: ~100 req/s
- p95 latency @ 100 users: ~1,000ms (10 users/pod)
- **Cost:** 5× infrastructure
- **Complexity:** Low (just scale replicas)

**Option 2: Vertical + Horizontal Scaling**
- Deploy 5 replicas with 2× CPU each
- Expected throughput: ~150 req/s
- p95 latency @ 100 users: ~600ms
- **Cost:** 5× infrastructure
- **Complexity:** Low (change resource limits)

**Option 3: GPU Acceleration + Horizontal**
- Deploy 3 GPU-enabled replicas
- Expected throughput: 600+ req/s
- p95 latency @ 100 users: <500ms
- **Cost:** 2-3× infrastructure (GPU premium)
- **Complexity:** Medium (GPU node pools, driver setup)

**Recommended:** Option 2 for immediate production, then migrate to Option 3 for optimal performance/cost.

---

## Conclusion

### Summary of Findings

✅ **Strengths:**
1. **Zero Errors:** 100% success rate across 320 requests
2. **Stable Service:** No crashes, restarts, or OOM events
3. **Predictable Behavior:** Throughput ceiling clearly identified
4. **Model Loading Working:** LSTM model served consistently

⚠️ **Limitations:**
1. **Concurrency Bottleneck:** 32-worker limit causes queuing above 10 users
2. **Latency Degradation:** 473% increase when scaling 10→50 users
3. **CPU Saturation:** Event loop lag suggests processing constraint
4. **Throughput Ceiling:** Cannot exceed ~21 req/s with current configuration

### Production Readiness Assessment

**Current State: Development/Testing ✅**
- Suitable for: <10 concurrent users
- Use case: Internal testing, demos, dev environments

**Production State: Requires Scaling ⚠️**
- Need: 5-10× replicas for 50-100 concurrent users
- Need: 2-4× CPU per pod for better latency
- Need: GPU acceleration for optimal performance

### Next Steps

1. **Immediate:** Deploy to production cluster with 5+ replicas
2. **Short-term:** Implement HPA for auto-scaling (3-20 replicas)
3. **Medium-term:** Vertical scaling with 2-4 CPU cores per pod
4. **Long-term:** Migrate to GPU nodes for 10-20× performance gain

**Test Again After Scaling:** Re-run this load test with 10 replicas to validate 100+ user capacity.

---

**Report Generated:** November 4, 2025  
**Test Duration:** ~17 seconds (excluding pauses)  
**Total Requests:** 320  
**Success Rate:** 100%  
**System Status:** ✅ **OPERATIONAL** (within documented limits)
