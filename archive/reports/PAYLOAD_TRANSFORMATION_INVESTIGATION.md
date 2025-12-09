# Payload Transformation Investigation Report

## Date: November 5, 2025
## Investigation: Tracing "time" → "ts" field transformation and timestamp corruption

---

## 🔍 Investigation Summary

**Objective:** Identify where payloads are transformed from "time" field with unique timestamps to "ts" field with duplicate "2018-02-06 00:00:00" timestamps.

**Method:**
1. Instrumented Locust payload generation and sending
2. Tested direct HTTP requests from Locust pod to Inference API
3. Captured logs at payload generation, sending, and reception stages

---

## ✅ Key Findings

### Finding 1: Locust GENERATES Correct Payloads

**Evidence:**
```
[LOCUST_GENERATE] Generated 16 timestamps
[LOCUST_GENERATE] Unique count: 16
[LOCUST_GENERATE] Sample: ['2025-11-05T18:24:34', '2025-11-05T18:25:34', '2025-11-05T18:26:34']
[LOCUST_GENERATE] Field name will be: 'time'
```

**Conclusion:** `_build_synthetic_predict_payload()` creates payloads with:
- Field name: **"time"** (not "ts")
- Timestamps: **Unique**, monotonic, current datetime
- Format: `"YYYY-MM-DDTHH:MM:SS"`

### Finding 2: Direct Requests Preserve Payloads End-to-End

**Test:** Sent Python requests directly from Locust pod → Inference API

**Sent Payload:**
```json
{
  "data": {
    "time": ["2025-11-05T10:00:00", "2025-11-05T10:01:00", "2025-11-05T10:02:00", ...]
  }
}
```

**Received at Inference (from logs):**
```
[PAYLOAD_RAW] body_len=623 preview={"data": {"time": ["2025-11-05T10:00:00", ...
[PAYLOAD_JSON] Keys=['time', 'down', 'up', ...]
[PAYLOAD_TS_RAW] column=time total_count=5 sample=['2025-11-05T10:00:00', ...]
[PAYLOAD_TS_BEFORE_PARSE] column=time total=5 unique_raw=5
```

**Conclusion:** ✅ **NO TRANSFORMATION** occurs when sending directly from Locust pod to Inference API.

### Finding 3: Locust Headless Test Used Cached Predictions

**Test:** Ran `locust --headless -u 2 -t 15s`

**Observed:**
```
[LOCUST_SENDING] seq=0 has_data=False
[LOCUST_SENDING] data_keys=[]
[LOCUST_SENDING] json_preview={}...
```

**Result:** Only empty payloads `{}` sent (cached prediction mode)

**Why:** `_predict_cache_enabled` likely True and `has_df` True, so Locust uses cached predictions instead of synthetic payloads.

---

## 🎯 Remaining Mystery: Where Does "ts" + "2018-02-06" Come From?

### Previous Evidence (From Earlier Sessions)

Inference API logs showed:
```
[PAYLOAD_PARSE] Received data keys: ['ts', 'down', 'up', ...]
[PAYLOAD_VALIDATION] Warning: Column 'ts' has 29 duplicate timestamps
[PAYLOAD_TS_RAW] column=ts total_count=30 sample=['2018-02-06 00:00:00', ...]
```

### Hypothesis Tree

#### ❌ RULED OUT:
1. **Inference API transformation** - Confirmed NOT modifying payloads
2. **Network layer corruption** - Direct HTTP test showed no corruption
3. **Locust generation bug** - Code generates "time" field with unique timestamps
4. **Pandas/JSON parsing** - Test payload with "time" parsed correctly

#### ✅ LIKELY SOURCES (Not Yet Tested):

1. **Kafka Claim-Check Pattern**
   - If inference consumes from Kafka `inference-data` topic
   - Messages might contain corrupted payloads from preprocessing step
   - Field name: "ts" (common in preprocessing output)
   - Timestamps: Historical dataset (2018-02-06 from ElBorn.csv)
   
   **Evidence Supporting:**
   - Inference deployment has `CONSUMER_TOPIC_0=inference-data`
   - Preprocessing outputs use "ts" field (not "time")
   - Dataset files contain Feb 2018 data

2. **Locust Payload Caching/Switching Mechanism**
   - `PREDICT_PAYLOAD_MODE=auto` might switch to cached mode
   - Cached payloads might come from previous runs or test files
   - `payload-valid.json` in workspace has "ts" field with 2018 dates
   
   **Evidence Supporting:**
   - `_should_use_cached_predicts()` function exists
   - `PREDICT_PAYLOAD_MODE=auto` in environment
   - Found `payload-valid.json` with matching corruption pattern

3. **Multi-Path Request Flow**
   - HTTP `/predict` works correctly
   - Kafka consumer path might process different payload format
   - Two separate code paths in `api_server.py` or `main.py`
   
   **Evidence Supporting:**
   - Inference container has both API server and Kafka consumer
   - `CONSUMER_TOPIC_0` and `CONSUMER_TOPIC_1` configured
   - Previous logs showed "ts" corruption but only during load tests

---

## 🔬 Diagnostic Evidence Collected

### 1. Locust Instrumentation

**Added Logging:**
- `[LOCUST_GENERATE]` - Timestamp generation trace
- `[LOCUST_SENDING]` - Pre-send payload inspection (field names, unique counts, samples)
- `[LOCUST_RESPONSE]` - Response status tracking

**Files Modified:**
- `locust/locustfile.py` - Enhanced `predict()` task and `_build_synthetic_predict_payload()`

**Image:** `locust:trace-payload`
**Deployment:** Updated `locust-master` and `locust-worker` with `DEBUG_LOCUST_PAYLOAD=1`

### 2. Inference Instrumentation (From Previous Session)

**Logging Coverage:**
- Raw HTTP body reception
- JSON deserialization
- DataFrame creation
- `pd.to_datetime()` conversion (before/after)
- Index assignment, sorting, timezone stripping
- Feature engineering transformations

**Image:** `inference:debug-trace`
**Environment:** `DEBUG_PAYLOAD_TRACE=1`

### 3. Test Results

| Test | Source | Field Name | Timestamps | Result |
|------|--------|------------|------------|--------|
| Direct HTTP from Locust pod | Python requests | "time" | 5 unique (2025-11-05) | ✅ Received intact |
| Locust headless test | Locust master | N/A | Empty payload {} | ✅ Cached prediction |
| (Previous session) Load test | Unknown | "ts" | 30 duplicates (2018-02-06) | ❌ Corrupted |

---

## 📋 Next Investigation Steps

### Step 1: Check Kafka Message Payloads ⚠️ HIGH PRIORITY

**Hypothesis:** Corruption occurs in Kafka messages consumed by inference service

**Action:**
```bash
# Check inference-data topic for message structure
kubectl exec <kafka-pod> -- kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic inference-data \
  --from-beginning \
  --max-messages 5

# Look for:
# - Field name: "ts" vs "time"
# - Timestamp values: 2018-02-06 pattern
# - Source: claim-check from preprocessing
```

**Expected:** Find messages with "ts" field and 2018 dates matching corruption pattern

### Step 2: Trace Inference Kafka Consumer Path

**Hypothesis:** Separate code path for Kafka-triggered inference vs HTTP API

**Action:**
```bash
# Check if inference/main.py has Kafka consumer
kubectl exec deployment/inference -- cat /app/main.py | head -100

# Enable Kafka consumer logging
kubectl set env deployment/inference DEBUG_KAFKA_CONSUMER=1

# Trigger preprocessing to generate Kafka messages
kubectl apply -f .kubernetes/preprocess-deployment.yaml

# Monitor inference logs for Kafka message processing
kubectl logs -f deployment/inference | grep -E "(KAFKA|CONSUMER|inference-data)"
```

**Expected:** Find separate payload processing path for Kafka messages

### Step 3: Inspect Preprocessing Output Format

**Hypothesis:** Preprocessing outputs "ts" field with historical timestamps

**Action:**
```bash
# Check preprocessing container code
kubectl logs -l io.kompose.service=preprocess --tail=100 | grep -E "(ts|time|timestamp)"

# Check MinIO for preprocessed data format
kubectl exec <minio-pod> -- mc cat local/processed-data/processed_data.parquet | head -500

# Look for claim-check messages published
kubectl logs -l io.kompose.service=preprocess | grep -E "(publish|kafka|claim)"
```

**Expected:** Find "ts" field in preprocessing output and Kafka claims

### Step 4: Test Full Pipeline Flow

**Hypothesis:** Load test triggers Kafka claim-check → Inference consumer → Corruption

**Action:**
```bash
# 1. Trigger preprocessing
kubectl apply -f .kubernetes/preprocess-deployment.yaml

# 2. Monitor Kafka messages
kubectl logs <kafka-pod> --follow &

# 3. Monitor inference consumer logs
kubectl logs -f deployment/inference | grep -E "(PAYLOAD|KAFKA|CONSUMER)"

# 4. Check for "ts" field corruption in inference logs
```

**Expected:** See "ts" field with 2018 timestamps when processing Kafka claims

### Step 5: Compare HTTP vs Kafka Code Paths

**Action:**
```bash
# Find entry points
kubectl exec deployment/inference -- grep -n "def predict" /app/api_server.py
kubectl exec deployment/inference -- grep -n "kafka|consumer|subscribe" /app/main.py

# Compare DataFrame preparation
# - HTTP: _prepare_dataframe_for_inference()
# - Kafka: Unknown function (to be found)
```

---

## 🎯 Working Hypothesis

**Most Likely Root Cause:**

The inference service has **TWO SEPARATE DATA INGESTION PATHS**:

### Path 1: HTTP API `/predict` ✅ WORKING
- Source: Locust HTTP POST requests
- Field name: "time"
- Timestamps: Current, unique
- Processing: `_prepare_dataframe_for_inference()`
- Result: **NO CORRUPTION**

### Path 2: Kafka Consumer ❌ SUSPECTED BROKEN
- Source: Kafka topic `inference-data`
- Messages: Claim-check from preprocessing
- Field name: "ts" (preprocessing convention)
- Timestamps: Historical dataset (2018-02-06)
- Processing: Unknown function (to be investigated)
- Result: **CORRUPTION OCCURS HERE**

**Evidence Supporting This Theory:**
1. Inference deployment configured with Kafka consumer (`CONSUMER_TOPIC_0=inference-data`)
2. HTTP tests show NO corruption
3. Load tests (which might trigger full pipeline) show corruption
4. Field name "ts" matches preprocessing conventions
5. Timestamps "2018-02-06" match dataset date range

**To Confirm:**
Check Kafka messages and trace inference consumer code path.

---

## 📊 Code Instrumentation Status

### ✅ Completed
- Inference HTTP API fully traced (20+ checkpoints)
- Locust payload generation traced
- Locust HTTP sending traced
- Direct HTTP test path validated

### ⏳ Remaining
- Kafka message content inspection
- Inference Kafka consumer instrumentation
- Preprocessing output format verification
- Full pipeline flow tracing

---

## 🔧 Environment Configuration

**Inference:**
- Image: `inference:debug-trace`
- Env: `DEBUG_PAYLOAD_TRACE=1`
- Replicas: 2
- Consumers: `inference-data`, `model-training`

**Locust:**
- Image: `locust:trace-payload`
- Env: `DEBUG_LOCUST_PAYLOAD=1`, `LOCUST_ALWAYS_LOG_FIRST=1`
- Master: 1 pod
- Workers: 4 pods

**Collection Commands:**
```bash
# Inference logs
kubectl logs -l app=inference --tail=200 | grep "PAYLOAD"

# Locust logs
kubectl logs deployment/locust-master | grep "LOCUST"

# Kafka messages
kubectl exec <kafka-pod> -- kafka-console-consumer --topic inference-data --max-messages 5
```

---

## 📄 Related Files

- `inference_container/api_server.py` - HTTP API (instrumented)
- `inference_container/data_utils.py` - Data transformations (instrumented)
- `inference_container/main.py` - Kafka consumer (NOT YET CHECKED)
- `locust/locustfile.py` - Load test harness (instrumented)
- `preprocess_container/main.py` - Data preprocessing (NOT YET CHECKED)
- `payload-valid.json` - Sample payload with "ts" + 2018 dates

---

## ✅ Conclusion

**HTTP Path: VALIDATED**
- Payloads flow intact from Locust → Inference API
- Field names preserved
- Timestamps unique and current
- NO corruption detected

**Kafka Path: UNVALIDATED** ⚠️
- Suspected source of corruption
- Next investigation focus
- Requires Kafka message inspection

**Recommendation:** Investigate Kafka consumer path in inference service as primary suspect for "ts" field and "2018-02-06" timestamp corruption.
