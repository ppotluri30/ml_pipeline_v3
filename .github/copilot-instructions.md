
## FLTS ML Pipeline - AI Agent Guide

**Purpose:** Get AI coding agents productive quickly with architecture patterns, critical workflows, and project-specific conventions.

---

### Architecture Overview

**Claim-check ML pipeline:** Event-driven microservices communicating via Kafka topics with data artifacts stored in MinIO (S3-compatible). MLflow tracks experiments and serves as model registry.

**Pipeline flow:**
1. `preprocess_container` → reads raw CSV, applies transformations, writes Parquet to MinIO, publishes claim-check messages
2. `train_container` (GRU/LSTM/Prophet variants) → consumes claims, trains models, logs to MLflow, publishes training events  
3. `eval_container` → waits for all expected model types, scores them, writes promotion pointers, broadcasts selection
4. `inference_container` → loads promoted models, serves predictions via HTTP, logs results to MinIO

**Supporting services:**
- Kafka (message broker), MinIO (object storage), Postgres (MLflow backend), fastapi-app (MinIO gateway)
- Deployment options: Docker Compose (local), Kubernetes/Helm (production)

---

### Key File Map

| File | Critical Concepts |
|------|-------------------|
| `preprocess_container/main.py` | `build_active_config()` generates deterministic config hash; idempotency via `.meta.json` sidecars; `FORCE_REPROCESS=1` override |
| `train_container/main.py` | MLflow artifact structure: model weights under `<MODEL_TYPE>/`, scaler under `scaler/*.pkl`; duplicate training guard via `SKIP_DUPLICATE_CONFIGS` + `DUP_CACHE_MAX` |
| `eval_container/main.py` | Waits for `EXPECTED_MODEL_TYPES` per config_hash; composite scoring (weights in `SCORE_WEIGHTS`); promotion pointer writes to `model-promotion/<scope>/current.json` |
| `inference_container/main.py` | Pointer resolution cascade: `current.json` → `global/current.json` → `<identifier>/current.json`; scaler auto-discovery |
| `inference_container/inferencer.py` | Model loading, scaler resolution, prediction de-duplication via `_emitted_prediction_keys` |
| `locust/locustfile.py` | Load testing harness; Kafka burst seeding, predict warm-up, result logging to JSONL |

---

### Data Flow & Kafka Topics

**Kafka message contracts (exact schemas):**

```python
# training-data (from preprocess)
{"bucket": "processed-data", "object_key": "processed_data.parquet", 
 "config_hash": "abc123...", "identifier": "default", "v": 1, "size": 50000}

# model-training (from trainers)
{"operation": "Trained: GRU", "status": "SUCCESS", 
 "run_id": "mlflow-run-id", "config_hash": "abc123...", "identifier": "default"}

# model-selected (from eval)
{"model_uri": "runs:/run-id/GRU", "score": 0.042, "config_hash": "abc123...",
 "identifier": "default", "model_type": "GRU", "rmse": 1.23, ...}
```

**MinIO bucket conventions:**
- `processed-data/`: Parquet files + `.meta.json` sidecars
- `mlflow/`: MLflow artifact root  
- `model-promotion/`: Promotion history (`<identifier|global>/<config_hash>/promotion-*.json`) + current pointers (`current.json`)
- `inference-txt-logs/`: JSONL prediction logs

---

### Configuration & Idempotency System

**Config hash generation:** `build_active_config()` in `preprocess_container/main.py` creates SHA256 from:
- Environment toggle flags (HANDLE_NANS, CLIP_ENABLE, TIME_FEATURES_ENABLE, etc.)
- Data preprocessing params (NANS_THRESHOLD, LAGS_N, SCALER type)
- Optional `EXTRA_HASH_SALT` for forced differentiation

**Idempotency mechanism:**
1. Config hash embedded in Parquet metadata + `.meta.json` sidecar
2. On startup, preprocess checks existing `.meta.json` for matching hash
3. If match found and `FORCE_REPROCESS != 1`, skip processing and re-emit claim checks
4. Trainers maintain `(MODEL_TYPE, config_hash)` cache to skip duplicate training (controlled by `SKIP_DUPLICATE_CONFIGS`)

**Key environment variables:**
- `FORCE_REPROCESS=1` - bypass idempotency in preprocess
- `SKIP_DUPLICATE_CONFIGS=1` (default) - enable duplicate-train guard
- `DUP_CACHE_MAX=500` - max entries in duplicate cache
- `EXPECTED_MODEL_TYPES=GRU,LSTM,PROPHET` - models to wait for in eval

---

### Development Workflows

**Local development (Docker Compose):**

```powershell
# Full pipeline startup
docker compose up -d kafka minio postgres mlflow fastapi-app preprocess train_gru train_lstm nonml_prophet eval inference

# Watch specific service logs
docker compose logs -f train_gru

# Quick inference smoke test
docker run --rm --network flts-main_app-network curlimages/curl:8.10.1 -s -X POST http://inference:8000/predict -H "Content-Type: application/json" -d "{}"

# Access services
# MLflow UI: http://localhost:5000
# MinIO console: http://localhost:9001 (minioadmin/minioadmin)
# Inference API: http://localhost:8000
```

**Load testing (Locust):**

```powershell
# Start Locust UI + workers
docker compose up -d locust
docker compose up -d --scale locust-worker=4 locust-worker

# Automated matrix tests (run_all_locust_tests.ps1)
.\run_all_locust_tests.ps1 -TestDuration 60 -ReplicaCounts @(1,2,4) -WorkerCounts @(4) -UserCounts @(50,100)

# Results written to: locust/results/auto_matrix/auto_summary.csv
```

**Kubernetes deployment:**

```bash
# Development (minimal resources)
helm install flts .helm/ -f .helm/values-complete.yaml -f .helm/values-dev.yaml

# Production (HA, autoscaling)
helm install flts-prod .helm/ -f .helm/values-complete.yaml -f .helm/values-prod.yaml --namespace flts-prod --create-namespace

# Check HPA status
kubectl get hpa inference-hpa
kubectl describe hpa inference-hpa
```

**Testing:**
- Unit tests: `pytest inference_container/tests/test_sync_predict.py` (install deps: `pip install -r inference_container/requirements.txt`)
- Readiness probes: `/readyz` (preprocess, eval), `/healthz` (all services)
- Metrics: Prometheus format at `http://inference:8000/metrics`

---

### Critical Patterns & Conventions

**MLflow artifact layout (trainers MUST follow):**
```
runs/<run-id>/
  artifacts/
    <MODEL_TYPE>/          # e.g., GRU/, LSTM/, PROPHET/
      model_weights.pth    # PyTorch models
      model.pkl            # Prophet/statsforecast models
    scaler/
      scaler.pkl           # REQUIRED - missing causes scaler_not_found error
```

**Promotion pointer resolution (inference):**
1. Try root-level: `model-promotion/current.json`  
2. Fallback to global: `model-promotion/global/current.json`
3. Fallback to identifier-scoped: `model-promotion/<identifier>/current.json`
4. Load model via `model_uri` from pointer, discover scaler via MLflow artifact listing

**Kafka consumer groups:** Each trainer service MUST have unique `CONSUMER_GROUP_ID` to receive all messages:
- `train-gru`: `CONSUMER_GROUP_ID=train-gru`
- `train-lstm`: `CONSUMER_GROUP_ID=train-lstm`  
- `nonml-prophet`: `CONSUMER_GROUP_ID=nonml-prophet`

**Structured logging keys (DO NOT RENAME - used by dashboards):**
- `skip_idempotent` (preprocess idempotency hit)
- `train_success_publish` (training completion)
- `promotion_scoreboard` (eval scoring results)
- `promotion_artifacts_ok` (promotion validation)
- `queue_enqueued`, `predict_inference_start` (inference metrics)

---

### Common Pitfalls

1. **Missing scaler artifact** → Inference fails with `scaler_not_found`. Ensure trainers log scaler to `scaler/` artifact path.
2. **Bucket name mismatches** → Silent upload failures. Run `_ensure_buckets()` in trainers; verify `INFERENCE_LOG_BUCKET` consistency.
3. **Promotion pointer schema changes** → Breaks pointer resolution. Maintain `model_uri`, `run_id`, `model_type`, `config_hash` keys.
4. **Adding new model types** → Update `EXPECTED_MODEL_TYPES`, add Compose service with unique `CONSUMER_GROUP_ID`, verify eval waits correctly.
5. **Windows PowerShell CRLF issues** → Locust commands using `/bin/sh -c` conditionals fail; use command arrays instead (see `BACKPRESSURE_NOTES.md`).

---

### Quick Reference

**Force full pipeline rerun:**
```powershell
$env:FORCE_REPROCESS = "1"
docker compose up -d preprocess
# Wait for completion, then restart trainers
docker compose restart train_gru train_lstm nonml_prophet
```

**Check model promotion status:**
```bash
# Via MinIO CLI (inside container)
docker exec -it minio mc ls myminio/model-promotion/current.json

# Via gateway
curl http://localhost:8000/download/model-promotion/current.json
```

**Scale inference horizontally:**
```powershell
docker compose up -d --scale inference=4 inference
# Or in Kubernetes: kubectl scale deployment inference --replicas=4
```

---

**Related docs:** `README.md` (detailed setup), `BACKPRESSURE_NOTES.md` (load testing), `HPA_TESTING_GUIDE.md` (K8s autoscaling), `.helm/README.md` (Helm deployment)

Update this file when bucket schemas, promotion contracts, or critical env variables change.
