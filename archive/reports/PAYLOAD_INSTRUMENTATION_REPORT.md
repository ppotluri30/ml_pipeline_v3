# Inference Payload Lifecycle Instrumentation Report

## Date: November 5, 2025
## Issue: Timestamp corruption (2018-02-06 00:00:00) in Kubernetes but not Docker

---

## 🎯 Instrumentation Implemented

### 1. **Diagnostic Logging Framework**
Added comprehensive trace logging controlled by `DEBUG_PAYLOAD_TRACE=1` environment variable.

**Files Modified:**
- `inference_container/api_server.py`
- `inference_container/data_utils.py`

**Key Trace Points:**

#### A. Request Entry (`api_server.py` `/predict` endpoint)
- `[PAYLOAD_ENV]` - Environment snapshot (MODE, CACHE settings, pod hostname)
- `[PAYLOAD_RAW]` - Raw HTTP body (first 300 bytes)
- `[PAYLOAD_JSON]` - Deserialized JSON structure (keys, column count)

#### B. DataFrame Creation
- `[PAYLOAD_TS_RAW]` - Raw timestamp data before pandas processing
- `[PAYLOAD_DF]` - DataFrame shape after `pd.DataFrame(data)`
- `[PAYLOAD_TS_SEARCH]` - Timestamp column candidates

#### C. Timestamp Parsing
- `[PAYLOAD_TS_BEFORE_PARSE]` - **Before `pd.to_datetime()`**: unique count, sample values
- `[PAYLOAD_TS_AFTER_PARSE]` - **After `pd.to_datetime()`**: unique count, sample values
- `[PAYLOAD_TS_COLLAPSE]` - **❌ ALERT**: Triggered if unique count drops during parsing

#### D. Index Assignment & Transformations
- `[PAYLOAD_TS_ASSIGNED]` - Timestamp column selected as index
- `[PAYLOAD_TS_BEFORE_SORT]` - Before `df.sort_index()`
- `[PAYLOAD_TS_AFTER_SORT]` - After `df.sort_index()`
- `[PAYLOAD_TS_AFTER_STRIP]` - After `strip_timezones()`

#### E. Feature Engineering (`data_utils.py`)
- `[DATA_UTILS_STRIP_TZ_BEFORE]` - Before timezone removal
- `[DATA_UTILS_STRIP_TZ_AFTER]` - After timezone removal
- `[DATA_UTILS_STRIP_TZ_COLLAPSE]` - **❌ ALERT**: If unique count changes
- `[DATA_UTILS_TIME_TO_FEATURE_BEFORE]` - Before cyclical time features
- `[DATA_UTILS_TIME_TO_FEATURE_AFTER_COPY]` - After `df.copy()`
- `[DATA_UTILS_TIME_TO_FEATURE_AFTER_ASSIGN]` - After `.assign()` operations
- `[DATA_UTILS_TIME_TO_FEATURE_FINAL]` - Final output validation

#### F. Validation & Results
- `[PAYLOAD_TS_DEDUP_START]` - Deduplication triggered
- `[PAYLOAD_TS_DEDUP_END]` - Deduplication results
- `[PAYLOAD_TS_INSUFFICIENT]` - **❌ REJECTION**: < 2 unique timestamps
- `[PAYLOAD_TS_ZERO_SPACING]` - **❌ ALERT**: Zero-spaced duplicates detected
- `[PAYLOAD_BEFORE_FEATURE]` - DataFrame state before feature engineering
- `[PAYLOAD_AFTER_FEATURE]` - DataFrame state after feature engineering

---

## 🧪 Test Results

### Test 1: Unique "ts" Timestamps (2025-11-05 format)
**Payload:** 10 unique timestamps with 1-minute intervals
**Pod:** `inference-86cbb9b55d-nbxq8`
**Request ID:** `d7c7e83f`

**Trace Evidence:**
```
[PAYLOAD_TS_RAW] column=ts total_count=10 sample=['2025-11-05 10:00:00', '2025-11-05 10:01:00', ...]
[PAYLOAD_TS_BEFORE_PARSE] column=ts total=10 unique_raw=10
[PAYLOAD_TS_AFTER_PARSE] column=ts unique_after_parse=10
[PAYLOAD_TS_ASSIGNED] Using column=ts unique=10
[PAYLOAD_TS_BEFORE_SORT] unique=10 len=10
[PAYLOAD_TS_AFTER_SORT] unique=10
[DATA_UTILS_STRIP_TZ_BEFORE] unique=10 has_tz=True
[DATA_UTILS_STRIP_TZ_AFTER] unique=10
[PAYLOAD_TS_AFTER_STRIP] unique=10
[PAYLOAD_BEFORE_FEATURE] unique_ts=10 rows=10 cols=11
[DATA_UTILS_TIME_TO_FEATURE_BEFORE] unique=10 rows=10
[DATA_UTILS_TIME_TO_FEATURE_AFTER_COPY] unique=10
[DATA_UTILS_TIME_TO_FEATURE_AFTER_ASSIGN] unique=10
[DATA_UTILS_TIME_TO_FEATURE_FINAL] unique=10 rows=10 cols=17
[PAYLOAD_AFTER_FEATURE] unique_ts=10 rows=10 cols=17
[PAYLOAD_PARSE] Successfully prepared DataFrame: 10 rows, 17 features
```

**✅ FINDING: NO TIMESTAMP CORRUPTION DETECTED**
- All 10 unique timestamps preserved throughout entire pipeline
- No collapse during `pd.to_datetime()`, sorting, timezone stripping, or feature engineering
- Payload processing completed successfully
- Resulted in HTTP 500 (inference execution error, not parsing error)

### Test 2: Cached Prediction
**Status:** HTTP 200
**Result:** SUCCESS
**Trace:**
```
[DATA_UTILS_TIME_TO_FEATURE_BEFORE] unique=1 rows=1 sample=[Timestamp('2018-02-28 02:54:00')]
[DATA_UTILS_TIME_TO_FEATURE_FINAL] unique=1 rows=1 cols=17
```

---

## 📊 Key Observations

### 1. **Payload Parsing Pipeline is NOT the Source of Corruption**
- Test payload with unique "ts" timestamps maintained 100% uniqueness through:
  - JSON deserialization ✅
  - DataFrame creation ✅
  - `pd.to_datetime()` conversion ✅
  - Index assignment ✅
  - Sorting ✅
  - Timezone stripping ✅
  - Feature engineering ✅

### 2. **Instrumentation is Working Correctly**
- All trace points fire in correct sequence
- Pod hostname captured: `inference-86cbb9b55d-nbxq8`
- Request IDs tracked: `d7c7e83f`, `7be1be09`
- Environment variables confirmed: `DEBUG_PAYLOAD_TRACE=1`, `MODE=auto`, `CACHE=0`

### 3. **The "2018-02-06 00:00:00" Corruption Occurs BEFORE the API**
Based on previous observations where payloads arrive with:
- Field name already transformed to "ts" (not "time")
- All 30 timestamps identical
- Date from historical dataset (2018-02-06)

This suggests the corruption happens in:
- **Locust payload generation/caching mechanism**
- **Kafka message payload** (if using claim-check pattern)
- **Gateway/proxy layer** (if requests are routed through FastAPI gateway)

---

## 🔍 Next Diagnostic Steps

### Step 1: Instrument Locust Payload Generation
**Goal:** Verify what Locust **actually sends** vs what the API receives

**Action:**
```python
# In locust/locustfile.py _build_synthetic_predict_payload()
# Add logging before return:
print(f"[LOCUST_PAYLOAD_GENERATED] timestamp_field='time' unique_count={len(set(times))} sample={times[:3]}")
```

### Step 2: Check for Cached/Stale Payloads
**Files to inspect:**
- `payload-valid.json` (contains "ts" field with 2018 timestamps)
- Any `.cache` or `.pkl` files in Locust container
- Environment variable `PREDICT_PAYLOAD_MODE=<value>`

**Questions:**
- Is Locust using a cached payload file instead of generating synthetic data?
- Is there a payload serialization/deserialization step that corrupts timestamps?

### Step 3: Add Locust-Side Logging
**Goal:** Log outgoing request body in Locust

**Action:**
```python
# In locust/locustfile.py task methods
payload = _next_predict_payload()
print(f"[LOCUST_SENDING] Keys={list(payload.get('data', {}).keys())} ts_sample={payload['data'].get('time', payload['data'].get('ts', []))[:3]}")
response = self.client.post("/predict", json=payload)
```

### Step 4: Network Layer Inspection
**Goal:** Capture raw HTTP traffic between Locust and Inference

**Action:**
```bash
# Run tcpdump in Kubernetes
kubectl exec -it <locust-pod> -- tcpdump -i any -A 'tcp port 8000 and host inference'
```

### Step 5: Check Kafka Claims (if applicable)
**Goal:** Verify if inference-data topic contains corrupted payloads

**Action:**
```bash
kubectl exec -it <kafka-pod> -- kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic inference-data \
  --from-beginning --max-messages 5
```

---

## 🚨 Critical Questions to Answer

1. **Where does "time" become "ts"?**
   - Locust generates "time" field
   - API receives "ts" field
   - Transformation happens between Locust and API

2. **Where do unique timestamps become identical?**
   - Locust generates unique timestamps (evidence: code inspection)
   - API receives 30 identical "2018-02-06 00:00:00" values
   - Corruption happens before API ingestion

3. **Is this a Docker vs Kubernetes environment difference?**
   - Different payload sources?
   - Different Locust configurations?
   - Different caching behavior?

4. **Is the "2018-02-06" date significant?**
   - Matches dataset date range (ElBorn.csv, LesCorts.csv mention Feb 2018)
   - Suggests payload may come from actual dataset rather than synthetic generation

---

## 📝 Deployment Status

**Inference Deployment:**
- Image: `inference:debug-trace`
- Replicas: 2
- Pods: `inference-86cbb9b55d-kh7nc`, `inference-86cbb9b55d-nbxq8`
- Environment:
  - `DEBUG_PAYLOAD_TRACE=1` ✅
  - `PREDICT_PAYLOAD_MODE=auto`
  - `ENABLE_PREDICT_CACHE=0`
  - `HOSTNAME=inference-86cbb9b55d-<suffix>`

**Log Collection Command:**
```bash
kubectl logs -l app=inference --tail=200 | grep -E "(PAYLOAD_|DATA_UTILS_)"
```

**Test Command (Unique ts):**
```bash
kubectl run test-trace --rm -i --restart=Never --image=python:3.11-slim -- python -c "
import urllib.request, json
req = urllib.request.Request(
    'http://inference:8000/predict',
    data=json.dumps({
        'data': {
            'ts': ['2025-11-05 10:00:00', '2025-11-05 10:01:00', ...],  # 10 unique
            'down': [5000000]*10,
            ...
        },
        'inference_length': 1
    }).encode(),
    headers={'Content-Type': 'application/json'}
)
resp = urllib.request.urlopen(req)
print(resp.read().decode())
"
```

---

## ✅ Conclusion

**The inference API payload processing pipeline does NOT corrupt timestamps.**

Evidence:
- Test payload with 10 unique "ts" timestamps maintained 100% uniqueness through all transformation stages
- Every trace point shows `unique=10` from raw input to final feature dataframe
- No collapse detected in `pd.to_datetime()`, sorting, timezone stripping, or feature engineering

**The corruption occurs UPSTREAM of the inference API**, likely in:
1. Locust test harness (payload caching or generation logic)
2. Kafka message broker (if using claim-check pattern)
3. Gateway/proxy layer (if requests are intermediated)

**Next Action:** Instrument Locust payload generation and network layer to identify where "time" → "ts" transformation and timestamp duplication occurs.
