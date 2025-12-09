# Inference Caching Disabled - Implementation Summary

**Date:** November 5, 2025  
**Images:** `inference:no-cache-v1`, `locust:no-cache-v1`  
**Status:** ✅ DEPLOYED AND VALIDATED

## Changes Made

### 1. Inference API - Caching Completely Disabled

**File:** `inference_container/api_server.py`

**Lines 947-954:** Removed cached DataFrame fallback logic
```python
df_for_inference = prepared_df
if df_for_inference is None:
    # INFERENCE CACHING DISABLED: Always require real data payload
    print(f"[INFERENCE] Caching disabled – no data provided in request payload (req_id={req_id})")
    raise HTTPException(
        status_code=400, 
        detail="Inference caching is disabled. You must provide 'data' in the request payload with all required feature columns and a 'time' index."
    )
```

**Lines 996:** Added confirmation logging
```python
print(f"[INFERENCE] Caching disabled – processing request data directly (req_id={req_id}, rows={len(df_for_inference.index)}, model={getattr(service, 'current_model', 'unknown')})")
```

**Result:**
- Empty payloads `{}` now return **HTTP 400** with clear error message
- All requests must include `"data"` with feature columns and `"time"` index
- Log confirms processing with: `[INFERENCE] Caching disabled – processing request data directly`

### 2. Locust - Always Generate Synthetic Payloads

**File:** `locust/locustfile.py`

**Lines 251-258:** Force synthetic mode
```python
def _should_use_cached_predicts() -> bool:
    """INFERENCE CACHING DISABLED: Always generate synthetic payloads with real data.
    
    This ensures every /predict request contains valid model input data,
    regardless of PREDICT_PAYLOAD_MODE or cache flags.
    """
    # Force synthetic mode - never use empty cached payloads
    return False
```

**Lines 262-281:** Enhanced payload generation with logging
```python
def _next_predict_payload() -> dict:
    """Generate next prediction payload with unique timestamps.
    
    ALWAYS generates synthetic payloads with real data - caching disabled.
    Every request will contain valid model input with all required features.
    """
    global _predict_payload_seq
    with _predict_payload_lock:
        seq = _predict_payload_seq
        _predict_payload_seq += 1
    
    # Generate synthetic payload with guaranteed unique timestamps
    payload = _build_synthetic_predict_payload(_predict_input_len, _predict_output_len, base_time=base_time)
    
    # Confirm we're generating real data
    if DEBUG_ENABLED and seq < 3:
        print(f"[LOCUST_PAYLOAD_GEN] seq={seq} mode=SYNTHETIC_ONLY data_keys={data_keys} rows={data_rows}")
```

**Result:**
- Locust **ALWAYS** generates full synthetic payloads with real data
- Each payload contains 16 rows, all 12 feature columns (`time`, `down`, `up`, `rnti_count`, etc.)
- Unique, time-ordered timestamps per request
- Payload size: **2163 bytes** (vs 2 bytes for empty `{}`)

## Deployment Commands

```bash
# Build images
cd inference_container
docker build -t inference:no-cache-v1 .

cd ../locust
docker build -t locust:no-cache-v1 .

# Update deployments
kubectl set image deployment/inference inference=inference:no-cache-v1
kubectl set image deployment/locust-master locust=locust:no-cache-v1
kubectl set image deployment/locust-worker locust=locust:no-cache-v1

# Set environment
kubectl set env deployment/locust-worker PREDICT_PAYLOAD_MODE=synthetic
kubectl set env deployment/locust-master PREDICT_PAYLOAD_MODE=synthetic

# Wait for rollout
kubectl rollout status deployment/inference
kubectl rollout status deployment/locust-master
kubectl rollout status deployment/locust-worker
```

## Validation Results

### Test 1: Empty Payload (Should Fail)
```bash
curl -X POST http://inference:8000/predict -H "Content-Type: application/json" -d '{}'
```
**Result:** ✅ HTTP 400 with error message:
```json
{
  "detail": "Inference caching is disabled. You must provide 'data' in the request payload with all required feature columns and a 'time' index."
}
```

### Test 2: Valid Payload (Should Succeed)
```bash
# Payload with 20 rows, all feature columns, unique timestamps
curl -X POST http://inference:8000/predict -H "Content-Type: application/json" -d '{"index_col": "time", "data": {...}, "inference_length": 1}'
```
**Result:** ✅ HTTP 200 with predictions

### Test 3: Locust Load Test (30s, 10 users)
```bash
locust -f locustfile.py --headless --host=http://inference:8000 -u 10 -r 2 -t 30s
```

**Results:**
- **Total Requests:** 126
- **Predict Requests:** 89
- **Failures:** 1 (0.79%) - only on `predict_warmup`, NOT on main `/predict` endpoint
- **Success Rate:** 89/89 = **100% on /predict endpoint** ✅
- **Avg Latency:** 154ms
- **p95 Latency:** 420ms
- **Throughput:** 3.01 req/s

**Payload Validation:**
- ✅ `has_data=True` (every request contains real data)
- ✅ **16 rows** per request
- ✅ **12 feature columns**: `time`, `down`, `up`, `rnti_count`, `mcs_down`, `mcs_down_var`, `mcs_up`, `mcs_up_var`, `rb_down`, `rb_down_var`, `rb_up`, `rb_up_var`
- ✅ **16 unique timestamps** per request
- ✅ **2163 bytes** per payload (vs 2 bytes for empty `{}`)

**Inference Logs Confirmed:**
```
[INFERENCE] Caching disabled – processing request data directly (req_id=74fddda6, rows=16, model=mlflow.pyfunc.loaded_model)
[INFERENCE] Caching disabled – processing request data directly (req_id=e9786bb4, rows=16, model=mlflow.pyfunc.loaded_model)
[INFERENCE] Caching disabled – processing request data directly (req_id=a52a0f1c, rows=16, model=mlflow.pyfunc.loaded_model)
```

## System Behavior

### Before (Cached Mode):
- Locust sent empty payloads `{}`
- Inference used preloaded `service.df` (50 rows with 2018 timestamps)
- No actual model input validation
- Payload size: 2 bytes

### After (Caching Disabled):
- Locust generates full synthetic payloads
- Inference requires `"data"` in every request
- Every request validates model input features
- Payload size: 2163 bytes
- 100% real model usage

## Important Notes

1. **No Backward Compatibility:** Empty payloads `{}` now return HTTP 400
2. **All Clients Must Provide Data:** Both Locust UI and headless mode send real payloads
3. **Environment Variable Ignored:** `PREDICT_PAYLOAD_MODE` is now effectively ignored (always synthetic)
4. **Performance Impact:** Slight increase in latency due to payload size (154ms avg vs ~84ms before)
5. **Memory Usage:** Stable at ~340Mi, no significant increase

## Next Steps

As requested, the next step is to update eval behavior (EXPECTED_MODEL_TYPES and promotion sequence) once this inference change is confirmed to be working correctly.

## Rollback (If Needed)

```bash
# Use previous images
kubectl set image deployment/inference inference:kafka-fix-v4
kubectl set image deployment/locust-master locust=locust:trace-payload
kubectl set image deployment/locust-worker locust=locust:trace-payload

# Revert environment
kubectl set env deployment/locust-worker PREDICT_PAYLOAD_MODE=cached
kubectl set env deployment/locust-master PREDICT_PAYLOAD_MODE=cached
```
