# FLTS Time Series Pipeline

End-to-end, containerized time-series training and inference platform using a claim‑check pattern (Kafka + MinIO) and MLflow for experiment tracking / artifact registry.

## Architecture Overview

Pipeline stages:
1. Preprocess: Reads raw dataset, builds canonical config JSON, embeds metadata into processed Parquet, publishes *claim checks* to Kafka (`training-data`, optional `inference-data`).
2. Training (GRU, LSTM, Prophet, etc.): Consumes `training-data` messages, downloads Parquet via gateway, trains model, logs MLflow artifacts (model folder + scaler), emits `model-training` events (RUNNING then SUCCESS).
3. Evaluation/Promotion: Waits for all `EXPECTED_MODEL_TYPES` successes for a `config_hash`, scores them, writes promotion history under `model-promotion/<identifier|global>/<config_hash>/` and updates `current.json`, publishes `model-selected`.
4. Inference: Consumes `inference-data` (optional), `model-training` (fast path), and `model-selected` (promotion pointer) to load models, perform windowed inference, log JSONL results to `inference-logs`, and expose HTTP API.

Supporting services: MinIO (+ gateway fastapi-app), Kafka, Postgres-backed MLflow.

## Quick Start (Fresh Run)

```powershell
# 0. (Optional) Clean slate
docker compose down

# 1. Start core infrastructure
docker compose up -d kafka minio postgres mlflow fastapi-app

# 2. Run preprocessing (produces processed_data.parquet + test_processed_data.parquet)
docker compose up -d preprocess

# 3. Launch trainers (GRU, LSTM, Prophet baseline)
docker compose up -d train_gru train_lstm nonml_prophet

# 4. Start evaluator + inference services
docker compose up -d eval inference

# 5. (Optional) Watch logs
# Training progress
docker compose logs -f train_gru
# Evaluation & promotion
docker compose logs -f eval
# Inference model load & predictions
docker compose logs -f inference

# 6. Issue a test prediction once a model is promoted
# (Empty JSON fast-path if implemented)
docker run --rm --network flts-main_app-network curlimages/curl:8.10.1 -s -X POST http://inference:8000/predict -H "Content-Type: application/json" -d "{}"

# 7. View metrics
docker run --rm --network flts-main_app-network curlimages/curl:8.10.1 -s http://inference:8000/metrics
``` 

## End-to-End Pipeline & Locust Benchmarking

The sections below provide a straightforward, copy‑pasteable sequence of PowerShell commands (Windows PowerShell v5.1) to run the full pipeline from preprocess → train → eval → inference and to run distributed Locust benchmarks using the included automation script `run_all_locust_tests.ps1`.

Run all commands from the repository root (example path shown below):


Prerequisites
- Docker & Docker Compose (v2) installed and configured
- PowerShell (Windows PowerShell v5.1 is used in examples)
- The repo is checked out and you are in the project root

1) Export quick env vars (optional but recommended for full repeatable run)

```powershell
#$env variables are read by docker-compose where used; set FORCE_REPROCESS to force preprocessing
$env:SAMPLE_TRAIN_ROWS = ""
$env:SAMPLE_TEST_ROWS  = ""
$env:SAMPLE_STRATEGY   = ""
$env:SAMPLE_SEED       = ""
$env:FORCE_REPROCESS   = "1"
```

2) Start core infra (detached)

```powershell
docker compose up -d kafka minio postgres mlflow fastapi-app
```

3) Start preprocessing (detached) and watch logs until it completes or idempotent-skips

```powershell
docker compose up -d preprocess
docker compose logs -f preprocess
# wait for preprocess logs to show success/idempotent skip
```

4) Start training services (detached)

```powershell
docker compose up -d train_gru train_lstm nonml_prophet
docker compose logs -f train_gru
# use separate terminals for other trainers if desired
```

5) Start evaluation and allow it to promote the best model

```powershell
docker compose up -d eval
docker compose logs -f eval
# wait for promotion pointer under model-promotion/.../current.json
```

6) Start inference + load-balancer (detached) and confirm readiness

```powershell
docker compose up -d inference inference-lb
docker compose logs -f inference
Invoke-WebRequest -UseBasicParsing http://localhost:8000/readyz
Invoke-WebRequest -UseBasicParsing http://localhost:8023/healthz
```

7) Start Locust master + workers (keep containers persistent)

```powershell
# bring up master and at least a worker service
docker compose up -d locust locust-worker
# scale workers (example to 4)
docker compose up -d --scale locust-worker=4 locust-worker
Invoke-WebRequest -UseBasicParsing http://localhost:8089/  # Locust UI
```

8) Quick validation using the included runner (30s smoke test)

```powershell
# Runs one quick test: 2 inference replicas, 4 workers, 50 users, 30s duration
.\run_all_locust_tests.ps1 -TestDuration 30 -ReplicaCounts @(2) -WorkerCounts @(4) -UserCounts @(50)

# Inspect outputs
ls locust\results\auto_matrix
Get-Content locust\results\auto_matrix\auto_summary.csv -Raw
Get-Content locust\results\auto_matrix\auto_summary.md -Raw
```

9) Run the full automated benchmark matrix (example matrix and recommended run-length)

```powershell
# Full matrix example: replicas 1,2,4,8 ; workers 4,8 ; users 200,400,800 ; 60s per test
.\run_all_locust_tests.ps1 -TestDuration 60 -ReplicaCounts @(1,2,4,8) -WorkerCounts @(4,8) -UserCounts @(200,400,800)

# For longer/stabler runs, change TestDuration to 120 or 180 seconds
.\run_all_locust_tests.ps1 -TestDuration 120 -ReplicaCounts @(1,2,4,8) -WorkerCounts @(4,8) -UserCounts @(200,400,800)
```

10) Manual headless Locust alternative (single run)

```powershell
# Example: scale inference to 4 replicas, scale workers to 4, then command master to swarm
docker compose up -d --scale inference=4 inference inference-lb
docker compose up -d --scale locust-worker=4 locust-worker
Start-Sleep -Seconds 15
Invoke-RestMethod -Uri "http://localhost:8089/swarm" -Method Post -Body @{user_count=400; spawn_rate=40; host="http://inference-lb"}
# Stop when done
Invoke-RestMethod -Uri "http://localhost:8089/stop" -Method Post
```

Results location
- CSV and Markdown summaries from the automated runner are stored in:

```
locust\results\auto_matrix\auto_summary.csv
locust\results\auto_matrix\auto_summary.md
```

Notes & tips
- The runner script `run_all_locust_tests.ps1` will scale inference replicas and Locust workers automatically and keeps containers persistent (no recreate) by default.
- Monitor MLflow UI at http://localhost:5000 to track training runs and check that scalers and model artifacts are logged correctly.
- If promotion pointer references missing artifacts, check `model-promotion/` in MinIO and trainer logs.
- Use `docker compose logs -f <service>` in separate terminals for parallel monitoring.

Troubleshooting
- If a PowerShell script fails to parse, ensure the file encoding is UTF-8 without BOM and contains only ASCII-safe characters in output strings.
- If Locust workers don't connect, verify network and that master is reachable at :8089 and worker services are running.

--

*This section was added to provide explicit end-to-end run steps and the Locust automation usage.*

```

## One-Liner (After First Build)
Bring up everything (training will run as soon as preprocess publishes claim checks):
```powershell
docker compose up -d kafka minio postgres mlflow fastapi-app preprocess train_gru train_lstm nonml_prophet eval inference
```

## Key Artifacts & Buckets
- Processed data: `processed-data/processed_data.parquet`, `processed-data/test_processed_data.parquet`
- MLflow artifacts: `mlflow` bucket (model folder = `MODEL_TYPE`, scaler in `scaler/`)
- Promotion history: `model-promotion/<identifier|global>/<config_hash>/`
- Inference logs: `inference-logs/<identifier>/<YYYYMMDD>/results.jsonl`

## Important Kafka Topics
- `training-data`: claim checks from preprocessing
- `inference-data`: (optional) claim checks for inference
- `model-training`: training lifecycle events (RUNNING, SUCCESS)
- `model-selected`: promotion (evaluation) events
- DLQ pattern: `DLQ-<base_topic>`

## Config Hash & Idempotency
Preprocessing builds a canonical JSON from env + config file values (sampling, trims, etc.), sorted & compact → SHA256 `config_hash`. Matching hash + `FORCE_REPROCESS!=1` short-circuits recompute.

Force new lineage:
- `EXTRA_HASH_SALT=<anything>`
- Or `FORCE_REPROCESS=1` to bypass cache reuse.

## Useful Environment Variables (Selected)
| Variable | Stage | Description |
|----------|-------|-------------|
| SAMPLE_TRAIN_ROWS / SAMPLE_TEST_ROWS | preprocess | Row sampling for faster dev; influences hash |
| EXTRA_HASH_SALT | preprocess | Forces new hash without logic change |
| MODEL_TYPE | train | GRU, LSTM, PROPHET, etc. |
| INPUT_SEQ_LEN / OUTPUT_SEQ_LEN | train/infer | Sequence lengths (must align) |
| EXPECTED_MODEL_TYPES | eval | List of model types required before scoring/promotion |
| RUN_INFERENCE_ON_TRAIN_SUCCESS | inference | Fast path model load on training SUCCESS |
| QUEUE_WORKERS, QUEUE_MAXSIZE | inference | Deprecated (local queue removed; concurrency handled externally) |
| WAIT_FOR_MODEL / MODEL_WAIT_TIMEOUT | inference | Startup gating before predictions |

## Promotion Scoring (Eval)
Weighted composite: `0.5*rmse + 0.3*mae + 0.2*mse` (lower is better, tie → newer start_time). Writes `current.json` pointer.

## Inference Behavior
- Attempts to load promoted model (tries `runs:/<run_id>/<run_name>` then fallback `runs:/<run_id>/model`).
- Flexible scaler discovery (searches for `.pkl` under `scaler/` or root with 'scaler' in name).
- Duplicate prediction suppression via `(run_id, prediction_hash)` set.
- HTTP requests execute inference synchronously inside the FastAPI worker; scale concurrency by adding more application replicas or external load generators (e.g. Locust distributed workers).
- JSONL output includes metrics (MAE/MSE/RMSE, per-feature errors, sample points, step-wise MAE if available).

## Troubleshooting Quick Table
| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| promotion_model_load_fail_startup | Stale promotion pointer after clean DB | Wait for new promotion or delete `model-promotion/current.json` |
| scaler_not_found | Trainer didnt log scaler | Confirm `mlflow.log_artifact(..., artifact_path="scaler")` |
| feature_count_mismatch | Preprocess change not in hash | Add env to canonical config or bump `EXTRA_HASH_SALT` |
| Inference overload | Application instance saturated | Add additional inference replicas or scale external load generators |
| No promotion triggered | Missing one model SUCCESS | Check `model-training` logs for all EXPECTED_MODEL_TYPES |

## Fast Dev Loop Tips
```powershell
# Force fresh preprocess + re-train quickly
$env:EXTRA_HASH_SALT = "dev$(Get-Random)"; docker compose up -d preprocess
# Shorten training epochs (edit compose EPOCHS=3) for speed
```

## Optional Load Test
```powershell
docker compose up -d locust
# Then open http://localhost:8089 OR headless:
docker compose run --rm -e LOCUST_HOST=http://inference:8000 locust -f /mnt/locust/locustfile.py --headless -u 40 -r 4 -t 20s
```

## Cleanup
```powershell
docker compose down
# (Optional) remove volumes/artifacts if you want a cold start
# docker volume rm flts-main_minio_data
```

## Roadmap / Ideas
- Add automated config hash unit test.
- Persist scaler deterministically across all trainers.
- Add statsforecast deep model variants (TCN, TETS) gating in EXPECTED_MODEL_TYPES.
- Improve startup handling when promotion pointer references missing runs.

---
*Generated to document the current operational pipeline after removal of `preprocess_container_backup`.*
