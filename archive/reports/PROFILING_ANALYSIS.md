# Inference Service Performance Analysis

## Initial Architecture Review

### Critical Path (from code inspection):

1. **HTTP Request → FastAPI** (`api_server.py`)
   - Middleware logging
   - Body parsing (Pydantic validation)
   - Cache check (if enabled)

2. **Data Preparation** (`api_server.py:_prepare_dataframe_for_inference`)
   - JSON → DataFrame conversion
   - `time_to_feature()` for temporal features
   - Column validation

3. **Concurrency Control** 
   - Semaphore acquire (wait time tracked)
   - `PREDICT_MAX_CONCURRENCY` limit (default: CPU count)

4. **Inference Execution** (`inferencer.py:perform_inference`)
   - DataFrame copy (deep copy for thread safety!)
   - `check_uniform()` - timedelta calculation
   - `time_to_feature()` - AGAIN for prediction frame
   - Model branch selection (pytorch/prophet/statsforecast)
   - **PyTorch path**:
     - `window_data()` - sliding window preparation
     - Tensor conversion + GPU transfer
     - **Model.predict()** - actual inference (per-step loop!)
     - Inverse scaling
   - Save & publish results

5. **Response Construction**
   - DataFrame → JSON serialization
   - Metrics updates

---

## Suspected Bottlenecks (Priority Order)

### 🔴 CRITICAL - Likely Root Causes

1. **Per-Step Prediction Loop** (Line 525-650 in inferencer.py)
   - **Problem**: For `inference_length=30`, calls `model.predict()` 30 TIMES sequentially
   - **Impact**: If each call is 50-100ms, that's 1.5-3 seconds RIGHT THERE
   - **Evidence**: Code shows: `for step in range(local_inference_length): multi_step_pred = _timed_predict(data_np)`
   - **Fix**: Batch predictions or use model's native multi-step forecast

2. **Double `time_to_feature()` Calls**
   - **Problem**: Called TWICE - once in prep, once for prediction frame
   - **Impact**: Trigonometric calculations (sin/cos) for 6 features × rows
   - **Fix**: Cache or pre-compute

3. **Deep DataFrame Copies**
   - **Problem**: `df.copy(deep=True)` for thread safety (Line 372)
   - **Impact**: With 1000+ rows, this copies all data + metadata
   - **Fix**: Use shallow copy + immutable operations, or lock-free data structures

4. **check_uniform() Overhead**
   - **Problem**: Calculates timedelta from index every request
   - **Impact**: Pandas datetime operations on full index
   - **Fix**: Cache timedelta when data is loaded

### 🟡 MODERATE - Likely Contributors

5. **window_data() Computation**
   - Sliding window creation with NumPy
   - May be creating large intermediate arrays

6. **Tensor Operations**
   - `torch.from_numpy()` + `.to(device)` transfers
   - If using CPU, this is copying data
   - If using CUDA, this waits for GPU

7. **JSON Serialization**
   - DataFrame → dict → JSON for response
   - Pandas `.to_dict()` can be slow for large frames

### 🟢 MINOR - But Easy Wins

8. **Metrics Updates** (Prometheus)
   - Multiple histogram observations per request
   - Thread-safe counter increments with locks

9. **Structured Logging**
   - Multiple `print()` calls with dict formatting
   - JSON serialization on every log

---

## Measurement Strategy

### Phase 1: Add Granular Timers
Instrument these specific stages (already partially done in code):

```python
timings = {
    "prep_ms": 0,           # Data preparation
    "copy_ms": 0,           # DataFrame deep copy
    "check_uniform_ms": 0,  # Timedelta calculation  
    "window_data_ms": 0,    # Sliding windows
    "tensor_prep_ms": 0,    # NumPy→Tensor conversion
    "model_loop_ms": 0,     # TOTAL time in prediction loop
    "model_predict_ms": 0,  # CUMULATIVE model.predict() calls
    "model_predict_calls": 0,
    "inverse_scale_ms": 0,
    "save_publish_ms": 0,
    "serialize_ms": 0
}
```

### Phase 2: Run Quick Test
- 10 users, 30 seconds
- Capture timing breakdown from logs
- Calculate per-request averages

---

## Expected Findings

Based on code structure, I predict:

1. **Model loop dominates**: 60-70% of total latency
   - 30 calls × 50ms = 1500ms
   
2. **Data prep overhead**: 10-15%
   - Deep copies, time features, window_data

3. **Everything else**: 15-30%
   - Serialization, metrics, logging

---

## Optimization Roadmap

### Quick Wins (Target: 3-5x speedup)

1. **Batch Model Predictions**
   - Replace per-step loop with single batched call
   - Expected gain: 50-70% reduction in model time

2. **Cache Computed Features**
   - Store timedelta, pre-computed time features
   - Expected gain: 10-20% reduction overall

3. **Shallow DataFrame Copy**
   - Use `.copy(deep=False)` + immutable ops
   - Expected gain: 5-10% reduction

4. **Remove Duplicate time_to_feature()**
   - Call once, reuse results
   - Expected gain: 5-10% reduction

### Medium-Term (Target: 10x+ speedup)

5. **Model Compilation** (PyTorch)
   - Use `torch.compile()` or TorchScript
   - Expected gain: 2-3x on model execution

6. **Async I/O for Kafka/S3**
   - Non-blocking publish operations
   - Expected gain: Remove blocking overhead

7. **Multi-Worker Deployment**
   - Increase Uvicorn workers (currently appears to be 1)
   - Expected gain: Scale with CPU cores

### Long-Term (Target: 100x+ with infra changes)

8. **Model Server (Triton/TorchServe)**
   - Dedicated inference engine with batching
   - Expected gain: 10-50x with GPU batching

9. **GPU Acceleration**
   - Move to GPU instances
   - Expected gain: 5-20x for large models

---

## Next Steps

1. ✅ Add detailed timing instrumentation
2. ⏳ Run profiling test (10 users, 30s)
3. ⏳ Analyze timing breakdown
4. ⏳ Implement top 3 optimizations
5. ⏳ Re-test and validate improvements
