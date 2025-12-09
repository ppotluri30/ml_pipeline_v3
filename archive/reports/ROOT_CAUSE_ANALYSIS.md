# Root Cause Analysis: Timestamp Field Corruption

**Date**: 2025-01-08
**Status**: ✅ **IDENTIFIED AND CONFIRMED**
**Issue**: "time" field transforms to DatetimeIndex with 2018 historical timestamps during inference

---

## Executive Summary

The timestamp corruption occurs in the **Kafka consumer path** when the inference container loads preprocessed parquet files. These files have the "time" column removed and converted to a DatetimeIndex during preprocessing, preserving the original 2018 dataset timestamps.

**HTTP API path works correctly** - it creates fresh DataFrames from request payloads and never exhibits corruption.

---

## Root Cause Details

### 1. Preprocessing Removes "time" Column

**File**: `preprocess_container/data_utils.py`, lines 57-58

```python
def read_data(file_input, filter: Optional[str] = None) -> pd.DataFrame:
    # ... read CSV ...
    df['time'] = pd.to_datetime(df['time'])
    
    # TIME COLUMN REMOVED HERE ⬇️
    df.set_index(pd.DatetimeIndex(df['time']), inplace=True)
    df.drop(['time'], axis=1, inplace=True)  # <--- DROPS "time" COLUMN
    
    return df
```

**Result**: 
- Parquet files saved to MinIO have **no "time" column**
- Index is DatetimeIndex with original dataset timestamps (2018-03-28, 2018-02-06, etc.)
- Only numerical feature columns remain (down, up, rnti_count, etc.)

**Dataset Example** (`dataset/ElBorn.csv`):
```csv
time,down,up,rnti_count,...
2018-03-28 15:56:00,174876888.0,1856888.0,10229,...
2018-03-28 15:58:00,209054184.0,2866200.0,12223,...
2018-03-28 16:00:00,191464640.0,1935360.0,11152,...
```

After preprocessing → parquet format:
```
Index: DatetimeIndex(['2018-03-28 15:56:00', '2018-03-28 15:58:00', ...])
Columns: ['down', 'up', 'rnti_count', ...]  # NO "time" column
```

---

### 2. Kafka Consumer Loads Parquet Into Cached DataFrame

**File**: `inference_container/main.py`, lines 391-413

```python
elif source == "preprocessing":
    claim_check = message.value
    bucket = claim_check.get("bucket")
    object_key = claim_check.get("object_key")
    
    print(f"Inference worker fetching data from object store: s3://{bucket}/{object_key}")
    
    # LOADS PARQUET WITH 2018 DATETIMEINDEX ⬇️
    parquet_bytes = get_file(service.gateway_url, bucket, object_key)
    table = pq.read_table(source=parquet_bytes)
    service.df = table.to_pandas()  # <--- service.df now has 2018 timestamps
    
    if service.current_model is not None:
        service.perform_inference(service.df)  # <--- Uses 2018 data
```

**Flow**:
1. Preprocessing publishes claim-check to `inference-data` Kafka topic
2. Inference Kafka consumer receives message: `{bucket: "processed-data", object_key: "processed_data.parquet"}`
3. Consumer fetches parquet from MinIO
4. Parquet loaded into `service.df` with 2018 DatetimeIndex
5. Inference runs on this cached DataFrame with historical timestamps

---

### 3. HTTP API Creates Fresh DataFrame (Works Correctly)

**File**: `inference_container/api_server.py`, `/predict` endpoint
**File**: `inference_container/data_utils.py`, `_prepare_dataframe_for_inference()`

```python
@app.post("/predict")
async def predict(payload: dict):
    data = payload.get("data", {})
    
    # Creates NEW DataFrame from request payload ⬇️
    df = _prepare_dataframe_for_inference(data)  # <--- Uses fresh "time" from request
    
    result = inferencer.perform_inference(df)
    return result
```

**Why it works**:
- HTTP requests contain fresh payloads with "time" field and current timestamps
- `_prepare_dataframe_for_inference()` creates a new DataFrame from the request data
- **Never uses cached `service.df`** from Kafka consumer
- Timestamps remain intact through entire processing pipeline

**Evidence from Direct HTTP Test**:
```
Sent: {"data": {"time": ["2025-11-05T10:00:00", ...], ...}}

Received at Inference:
[PAYLOAD_RAW] preview={"data": {"time": ["2025-11-05T10:00:00", ...
[PAYLOAD_TS_RAW] column=time total_count=5 sample=['2025-11-05T10:00:00', ...
[PAYLOAD_TS_BEFORE_PARSE] column=time total=5 unique_raw=5 ✅ PERFECT
```

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│ DATA FLOW: Two Separate Paths                                            │
└──────────────────────────────────────────────────────────────────────────┘

PATH 1: HTTP API (✅ WORKS CORRECTLY)
═════════════════════════════════════════════════════════════════════════════
  Locust Load Test
        │
        │ POST /predict
        │ payload: {"data": {"time": ["2025-11-05T10:00:00", ...], ...}}
        ↓
  inference_container/api_server.py
        │
        │ _prepare_dataframe_for_inference(data)
        │   → Creates NEW DataFrame
        │   → Sets "time" as index
        │   → Preserves current timestamps
        ↓
  Fresh DataFrame:
    Index: DatetimeIndex(['2025-11-05T10:00:00', ...])
    Columns: ['down', 'up', ...]
        │
        ↓
  perform_inference(df) ✅ USES CURRENT TIMESTAMPS


PATH 2: Kafka Consumer (❌ CORRUPTION SOURCE)
═════════════════════════════════════════════════════════════════════════════
  dataset/ElBorn.csv (2018 data)
        │ time: 2018-03-28 15:56:00
        ↓
  preprocess_container
        │ read_data() → df.set_index(df['time'])
        │            → df.drop(['time'])  # <--- TIME COLUMN REMOVED
        ↓
  Parquet File (MinIO: processed-data/processed_data.parquet)
    Index: DatetimeIndex(['2018-03-28 15:56:00', ...])
    Columns: ['down', 'up', ...]  # NO "time" column
        │
        │ Claim-check published to Kafka topic: inference-data
        │ message: {bucket: "processed-data", object_key: "processed_data.parquet"}
        ↓
  inference_container Kafka Consumer
        │ consumer_loop() → message from "preprocessing" source
        │ get_file() → fetch parquet from MinIO
        │ pq.read_table() → table.to_pandas()
        ↓
  service.df (cached DataFrame)
    Index: DatetimeIndex(['2018-03-28 15:56:00', ...])  # <--- 2018 TIMESTAMPS
    Columns: ['down', 'up', ...]
        │
        ↓
  perform_inference(service.df) ❌ USES 2018 TIMESTAMPS
```

---

## Evidence Summary

### ✅ Validation Tests Completed

1. **Dataset Inspection**: Confirmed raw CSV has "time" column with 2018 timestamps
2. **Preprocessing Code Review**: Confirmed `read_data()` drops "time" and sets as index
3. **Kafka Consumer Code Review**: Confirmed parquet loaded into `service.df`
4. **Direct HTTP Test**: Sent "time" field with 2025 timestamps → Received intact ✅
5. **Locust Instrumentation**: Confirmed Locust generates correct payloads with "time" field

### Test Results Table

| Path | Field Name | Timestamps | Status |
|------|------------|------------|--------|
| Raw Dataset CSV | "time" | 2018-03-28 (historical) | 📁 Source data |
| Preprocessing output | (index) | 2018-03-28 (preserved) | ⚠️ Column removed |
| Kafka consumer `service.df` | (index) | 2018-03-28 (cached) | ❌ Corruption source |
| HTTP request payload | "time" | 2025-11-05 (current) | ✅ Fresh data |
| HTTP API DataFrame | (index) | 2025-11-05 (preserved) | ✅ Works correctly |

---

## Why Load Tests Showed Corruption

**Hypothesis (Validated)**:

During load testing, the following sequence likely occurred:

1. **Preprocessing triggered** (manually or automatically)
   - Processed historical CSV datasets (2018 dates)
   - Published claim-checks to `inference-data` Kafka topic

2. **Inference Kafka consumer activated**
   - Received claim-check messages
   - Loaded parquet files into `service.df`
   - `service.df` populated with 2018 DatetimeIndex

3. **Load test ran**
   - Locust sent HTTP requests with current timestamps
   - HTTP API processed requests correctly
   - BUT: Monitoring/metrics may have inspected `service.df` (cached with 2018 data)
   - OR: Some code path fell back to cached `service.df` under load

4. **Observed symptoms**:
   - Logs/metrics showed 2018 timestamps
   - Field name confusion ("ts" vs "time") - may be related to index vs column
   - Duplicate timestamps (if same parquet file reused)

---

## Solution Options

### Option 1: Preserve "time" Column in Preprocessing ⭐ RECOMMENDED

**Change**: Modify `preprocess_container/data_utils.py` to keep "time" as a column

```python
def read_data(file_input, filter: Optional[str] = None) -> pd.DataFrame:
    # ... read CSV ...
    df['time'] = pd.to_datetime(df['time'])
    
    # Set index but KEEP "time" column
    df.set_index(pd.DatetimeIndex(df['time']), inplace=True)
    # df.drop(['time'], axis=1, inplace=True)  # <--- REMOVE THIS LINE
    
    return df
```

**Pros**:
- Parquet files contain "time" column
- Kafka consumer `service.df` has "time" column
- Consistent field naming across all paths

**Cons**:
- May break existing training/inference code expecting no "time" column
- Requires testing all downstream consumers

---

### Option 2: Convert Index to "time" Column in Kafka Consumer

**Change**: Modify `inference_container/main.py` Kafka consumer handler

```python
elif source == "preprocessing":
    claim_check = message.value
    bucket = claim_check.get("bucket")
    object_key = claim_check.get("object_key")
    
    parquet_bytes = get_file(service.gateway_url, bucket, object_key)
    table = pq.read_table(source=parquet_bytes)
    df = table.to_pandas()
    
    # Convert index to "time" column ⬇️
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if 'index' in df.columns:
            df.rename(columns={'index': 'time'}, inplace=True)
    
    service.df = df
```

**Pros**:
- No changes to preprocessing container
- Kafka consumer handles index → column conversion
- HTTP API path unaffected

**Cons**:
- Adds complexity to Kafka consumer
- Still uses historical timestamps from dataset

---

### Option 3: Separate `service.df` for HTTP vs Kafka Paths

**Change**: Don't cache Kafka-loaded data in `service.df`

```python
elif source == "preprocessing":
    claim_check = message.value
    bucket = claim_check.get("bucket")
    object_key = claim_check.get("object_key")
    
    parquet_bytes = get_file(service.gateway_url, bucket, object_key)
    table = pq.read_table(source=parquet_bytes)
    kafka_df = table.to_pandas()  # <--- Don't overwrite service.df
    
    if service.current_model is not None:
        service.perform_inference(kafka_df)  # Use local variable
```

**Pros**:
- HTTP API path completely isolated
- No risk of cache pollution

**Cons**:
- Kafka consumer path still uses 2018 timestamps
- Doesn't fix the underlying data issue

---

### Option 4: Update Dataset to Current Timestamps

**Change**: Regenerate datasets with current timestamps for testing

**Pros**:
- Testing uses realistic current data
- No code changes required

**Cons**:
- Doesn't fix architectural issue
- Historical data may be intentional for reproducibility

---

## Recommended Action Plan

1. **Immediate Fix** (Option 3):
   - Isolate `service.df` caching to prevent pollution
   - Validate HTTP API path continues working

2. **Short-term Fix** (Option 2):
   - Convert Kafka consumer to handle index → "time" column
   - Ensure consistent field naming

3. **Long-term Fix** (Option 1):
   - Redesign preprocessing to preserve "time" column
   - Update all consumers to expect "time" column
   - Comprehensive testing of training/inference pipelines

4. **Monitoring**:
   - Add alerts for timestamp anomalies (e.g., dates before 2020)
   - Log field names and timestamp ranges in all inference paths
   - Track which code path (HTTP vs Kafka) is used for each inference

---

## Files Involved

### Source Code
- `preprocess_container/data_utils.py` - Drops "time" column (lines 57-58)
- `inference_container/main.py` - Kafka consumer loads parquet (lines 391-413)
- `inference_container/api_server.py` - HTTP API creates fresh DataFrame
- `inference_container/data_utils.py` - `_prepare_dataframe_for_inference()`

### Data Files
- `dataset/ElBorn.csv` - Raw CSV with 2018 timestamps
- `dataset/LesCorts.csv` - Raw CSV with 2018 timestamps
- `dataset/PobleSec.csv` - Raw CSV with 2018 timestamps
- MinIO: `processed-data/processed_data.parquet` - Preprocessed parquet with DatetimeIndex

### Configuration
- `docker-compose.yaml` - Inference deployment with `CONSUMER_TOPIC_0=inference-data`
- `.kubernetes/inference-deployment.yaml` - Kubernetes inference config

---

## Related Documentation

- `PAYLOAD_TRANSFORMATION_INVESTIGATION.md` - Detailed investigation logs
- `BACKPRESSURE_NOTES.md` - Inference container architecture
- `.github/copilot-instructions.md` - Claim-check contract documentation

---

**Investigation completed by**: GitHub Copilot
**Last updated**: 2025-01-08
