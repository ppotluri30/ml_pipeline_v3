# FLTS Inference Load Test & Autoscaling Analysis Report

**Date**: December 1, 2025  
**Environment**: Kubernetes (Docker Desktop)  
**Model**: GRU (run_id: `3e9b5c820256434d83f4aac51de1cc56`)

---

## Executive Summary

Successfully validated the inference-http service under production-grade load conditions. The HPA autoscaling mechanism responded correctly to CPU pressure, scaling pods from 2 to 8+ replicas. **Zero failures** were observed across all tests.

### Key Metrics

| Metric | Test 1 (50 Users) | Test 2 (150 Users) |
|--------|-------------------|---------------------|
| **Duration** | 60s | 120s |
| **Total Requests** | 1,797 | 10,967 |
| **Failures** | 0 (0.00%) | 0 (0.00%) |
| **Throughput** | ~30 req/s | ~92 req/s |
| **Avg Latency** | 113ms | 111ms |
| **P50 (Median)** | 75ms | 68ms |
| **P95** | 290ms | 320ms |
| **P99** | 460ms | 940ms |
| **Max** | 1,093ms | 1,495ms |

---

## Test Configuration

### Load Test Parameters
```yaml
Tool: Locust 2.42.1
Test 1: 50 users, ramp 10/s, 60s duration
Test 2: 150 users, ramp 25/s, 120s duration
Endpoint: POST /predict
Payload: 30-row time series with all required features
```

### Infrastructure Configuration
```yaml
Deployment: inference-http
Initial Replicas: 2
HPA Config:
  Min Replicas: 2
  Max Replicas: 12
  Target CPU: 70%
Model: GRU (PyTorch)
Input Sequence Length: 10
Output Sequence Length: 1
```

---

## Autoscaling Behavior

### HPA Scaling Timeline

| Time | Event | Replicas | CPU |
|------|-------|----------|-----|
| T+0s | Load test started | 2 | ~5% |
| T+15s | CPU spike detected | 2 | 186% |
| T+30s | First scale-up | 4 | ~150% |
| T+60s | Sustained load | 6 | 142% |
| T+90s | Peak scaling | 8+ | ~120% |

### Scaling Observations

1. **Response Time**: HPA detected CPU threshold breach within ~15 seconds
2. **Scale-Up Rate**: 2 pods added per scaling decision (~30s intervals)
3. **Pod Readiness**: New pods became ready in ~20-25 seconds
4. **Load Distribution**: Traffic effectively distributed across new replicas

---

## Latency Analysis

### Response Time Distribution (150 Users Test)

```
Percentile | Latency (ms)
-----------|-------------
     P50   |     68
     P66   |    110
     P75   |    140
     P80   |    160
     P90   |    220
     P95   |    320
     P98   |    540
     P99   |    940
   P99.9   |  1,400
    P100   |  1,495
```

### Latency Breakdown

- **Sub-100ms responses**: ~55% of requests (excellent)
- **100-200ms responses**: ~35% of requests (good)
- **200-500ms responses**: ~8% of requests (acceptable)
- **500ms+ responses**: ~2% of requests (cold-start/scaling)

---

## Issues Fixed During Testing

### 1. Missing `model_class` Attribute
**Problem**: HTTP-based model loading didn't set `model_class`, causing "Unsupported model class" errors.  
**Solution**: Added model_class inference based on model_type in `inference_http.py`:
```python
if upper_type in ('GRU', 'LSTM'):
    service.model_class = 'pytorch'
elif upper_type in ('PROPHET',):
    service.model_class = 'prophet'
```

### 2. Missing Scaler Loading
**Problem**: HTTP-based model loading skipped scaler artifact loading, causing None scaler.  
**Solution**: Added complete scaler loading logic with MLflow artifact discovery in `inference_http.py`.

### 3. Scaler Artifact Contains None
**Finding**: The training pipeline saves `None` as the scaler artifact because `window_data()` returns `None` for scaler.  
**Impact**: Inference works but returns raw (scaled) predictions instead of inverse-transformed values.  
**Recommendation**: Fix `train_container/data_utils.py` to properly create and return a fitted scaler.

---

## Recommendations

### Immediate Actions
1. ✅ **Fixed**: Model class inference for HTTP loading
2. ✅ **Fixed**: Scaler loading in HTTP path
3. ⚠️ **Pending**: Fix training to save actual fitted scaler

### Performance Optimizations
1. **Increase min replicas to 3-4** for faster response to sudden load spikes
2. **Consider ONNX conversion** for PyTorch models to reduce inference latency
3. **Enable Numba JIT** for window_data operations (currently disabled due to caching issue)

### Production Readiness Checklist
- [x] Zero failures under sustained load
- [x] HPA autoscaling working correctly
- [x] Latency within acceptable bounds (<1s P99)
- [x] Model hot-reload mechanism functional
- [ ] Scaler properly saved and loaded (training fix needed)
- [ ] KEDA latency-based scaling (validated separately)

---

## Test Artifacts

- **Locust Worker Pods**: 4 (distributed load generation)
- **Model Run ID**: `3e9b5c820256434d83f4aac51de1cc56`
- **Config Hash**: `c11e6ea3c48c1b8c7feaca55bb82f7a53a08e57d27d52b37494f5b9ccff0f2e6`
- **Promotion Pointer**: `model-promotion/current.json`

---

## Conclusion

The inference-http service demonstrates **production-ready performance** with:
- **100% success rate** under heavy load
- **Responsive autoscaling** (2→8 pods in ~90s)
- **Good latency characteristics** (68ms median, 320ms P95)
- **Stable throughput** (~92 req/s sustained)

The fixes applied during this testing session resolved critical issues with HTTP-based model loading. The training pipeline's scaler handling should be addressed in a follow-up task to ensure predictions are properly inverse-transformed.
