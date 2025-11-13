
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
- Monitoring stack: Prometheus (15s scrape), KEDA (15s polling), HPA v2 (CPU/latency/queue metrics)

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

**Real-time monitoring (Kubernetes):**

```powershell
# Query Prometheus p95 latency (15s scrape interval)
$query = 'histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[2m])) by (le))'
kubectl exec deployment/prometheus-server -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$([uri]::EscapeDataString($query))"

# Monitor HPA scaling in real-time
kubectl get hpa keda-hpa-inference-slo-scaler -w

# Run headless load test
kubectl exec deployment/locust-master -- locust --headless --host=http://inference:8000 -u 150 -r 10 -t 90s --print-stats

# Check KEDA ScaledObject status
kubectl get scaledobject inference-slo-scaler -o yaml
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/default/s0-prometheus"
```

**Automated HPA testing:**

```powershell
# Matrix test with autoscaling validation
.\scripts\k8s_auto_hpa_tests.ps1 -TestDuration 120 -UserCounts @(50,100,200) -WorkerCounts @(4,8) -HPATargetCPU 70

# Results: reports/k8s_hpa_performance/*.csv, *.md
```

**Testing:**
- Unit tests: `pytest inference_container/tests/test_sync_predict.py` (install deps: `pip install -r inference_container/requirements.txt`)
- Readiness probes: `/readyz` (preprocess, eval), `/healthz` (all services)
- Metrics: Prometheus format at `http://inference:8000/metrics` or `/prometheus`

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
6. **Prometheus NaN values** → Rate calculations need 2+ samples in query window. With 15s scrape interval, use `[2m]` window minimum (8-9 samples).
7. **KEDA latency threshold mismatches** → `threshold` vs `activationThreshold` in ScaledObject. `activationThreshold` prevents premature scale-down; set lower than `threshold`.
8. **HPA "Unknown" metrics** → Prometheus adapter not configured or custom metrics API unavailable. Verify: `kubectl get apiservice v1beta1.custom.metrics.k8s.io`.

---

### Monitoring & Autoscaling Configuration

**Critical timing parameters (validated 2025-11-12):**

```yaml
# Prometheus scrape (ConfigMap: prometheus-server)
scrape_interval: 15s       # Production validated - balances real-time visibility vs storage
scrape_timeout: 10s        # Safe for inference response times

# KEDA ScaledObject (inference-slo-scaler)
pollingInterval: 15        # Matches Prometheus scrape
cooldownPeriod: 300        # 5 minutes before scale-down
stabilizationWindowSeconds: 60  # Scale-up delay to prevent flapping

# Query windows for rate calculations
[2m] window                # Minimum for stable rate() with 15s scrape (8-9 samples)
[5m] window                # Preferred for production latency queries (20+ samples)
```

**KEDA latency trigger configuration:**

```yaml
triggers:
- type: prometheus
  metadata:
    serverAddress: http://prometheus-server.default.svc.cluster.local:80
    query: histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[5m])) by (le))
    threshold: "2.0"         # Scale up if p95 > 2.0s
    activationThreshold: "1.0"  # Activate scaler if p95 > 1.0s (prevents premature scale-down)
```

**Prometheus metric queries (PowerShell-safe escaping):**

```powershell
# P95 latency
$query = 'histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[2m])) by (le))'
kubectl exec deployment/prometheus-server -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$([uri]::EscapeDataString($query))"

# Average queue length
$query = 'avg(inference_queue_len)'
kubectl exec deployment/prometheus-server -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$([uri]::EscapeDataString($query))"
```

**HPA scaling expectations:**

```
0-15s:    Load ramp-up, CPU/latency increases
15-30s:   Metrics breach threshold, KEDA detects
30-60s:   HPA scaling decision + stabilization window
60-90s:   New pods starting, becoming ready
90-180s:  Load distributed, metrics stabilize
300s+:    Scale-down eligible after cooldown period
```

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

**Debug KEDA scaling issues:**
```powershell
# Check ScaledObject status
kubectl describe scaledobject inference-slo-scaler

# View KEDA operator logs
kubectl logs -n keda deployment/keda-operator -f

# Query external metrics API directly
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/default/s0-prometheus" | ConvertFrom-Json

# Verify Prometheus target health
kubectl exec deployment/prometheus-server -c prometheus-server -- wget -qO- 'http://localhost:9090/api/v1/targets' | ConvertFrom-Json | Select-Object -ExpandProperty data | Select-Object -ExpandProperty activeTargets | Where-Object { $_.labels.service -eq 'inference' }
```

**Access monitoring UIs (port-forward required):**
```powershell
# Prometheus
kubectl port-forward deployment/prometheus-server 9090:9090
# Access: http://localhost:9090

# Grafana
kubectl port-forward svc/grafana 3000:3000
# Access: http://localhost:3000

# Locust load testing UI
kubectl port-forward svc/locust-master 8089:8089
# Access: http://localhost:8089

# MLflow tracking
kubectl port-forward svc/mlflow 5000:5000
# Access: http://localhost:5000
```

---

**Related docs:** `README.md` (detailed setup), `BACKPRESSURE_NOTES.md` (load testing), `HPA_TESTING_GUIDE.md` (K8s autoscaling), `.helm/README.md` (Helm deployment), `REALTIME_MONITORING_VALIDATION.md` (Prometheus tuning)

**PowerShell automation scripts:**
- `.\scripts\k8s_auto_hpa_tests.ps1` - Automated HPA matrix testing with CSV/Markdown reporting
- `.\Monitor-LiveLatency.ps1` - Real-time latency monitoring during load tests (requires port-forward to Locust UI)
- `.\monitor_keda_scaling.ps1` - KEDA scaling event tracker with Prometheus queries
- `.\run_all_locust_tests.ps1` - Docker Compose load test matrix automation

Update this file when bucket schemas, promotion contracts, critical env variables, or monitoring configurations change.
