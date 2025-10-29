
## FLTS AI Agent Guide (Concise, Actionable)

Purpose: get an AI coding agent productive quickly — architecture, contracts, dev workflows, and repo-specific conventions.

- Architecture snapshot: claim-check ML pipeline: `preprocess_container` → `train_container` (GRU/LSTM/Prophet) → `eval_container` → `inference_container`. Components talk via Kafka topics and MinIO; a FastAPI gateway exposes `/download`, `/upload`, `/predict`, `/metrics`.

- Quick code tour (key files):
	- `preprocess_container/main.py` — `build_active_config()` (config-hash), parquet metadata stamps, idempotent short-circuit.
	- `train_container/main.py` — `_train_parquet` targets, MLflow artifact layout: model folder named by `MODEL_TYPE`, scaler under `scaler/*.pkl`.
	- `eval_container/main.py` — waits for `EXPECTED_MODEL_TYPES`, computes composite score (weights in code), writes promotion pointer JSONs.
	- `inference_container/main.py`, `inferencer.py` — pointer resolution (`current.json` → `global/current.json` → `<identifier>/current.json`), scaler discovery and loading, de-dup logic.
	- `locust/locustfile.py` — load harness and Kafka seeding knobs.

- Claim-check contract (concrete):
	- Preprocess emits claim messages to `training-data` with {bucket, object_key, config_hash, identifier}. Some flows also publish `inference-data`.
	- Trainers publish `model-training` events (only `SUCCESS` after artifacts logged) including `run_id`, `experiment`, `config_hash`.
	- Eval publishes promotion payloads (`model_uri`, `score`) and writes pointers under `model-promotion/<identifier|global>/<config_hash>/current.json` before broadcasting `model-selected`.

- Important runtime and config conventions:
	- Config hash: `build_active_config()` folds env toggles, `_data` payload, and optional `EXTRA_HASH_SALT`. Hash is embedded in parquet metadata and `.meta.json` claim files.
	- Idempotency: matching config hash skips processing unless `FORCE_REPROCESS=1`.
	- Duplicate-train guard: trainers use a `(MODEL_TYPE, config_hash)` cache controlled by `SKIP_DUPLICATE_CONFIGS` and `DUP_CACHE_MAX`.
	- Buckets must be consistent: `_ensure_buckets()` must align training and inference `INFERENCE_LOG_BUCKET` or inference uploads fail silently.
	- MLflow expectation: scaler artifact under `scaler/*.pkl` — missing scaler causes `scaler_not_found` at inference load time.

- How to run locally (fast checks):
	- Full fast loop (images prebuilt):
		docker compose up -d kafka minio postgres mlflow fastapi-app preprocess train_gru train_lstm nonml_prophet eval inference
	- Single-service smoke for inference: `docker compose up -d inference` — health and endpoints at `/predict`, `/metrics`, `/scale_workers` (see `inference_container/api_server.py`).
	- Load smoke (example): `docker compose run --rm -e LOCUST_HOST=http://inference:8000 locust -f /mnt/locust/locustfile.py --headless -u 40 -r 4 -t 20s`.

- Tests and diagnostics:
	- Unit/smoke test: `pytest inference_container/tests/test_backpressure.py` exercises predict path and backpressure/cache behavior. Install deps: `pip install -r inference_container/requirements.txt`.
	- Readiness endpoints: preprocess `/readyz`, eval `/readyz`. Prometheus metrics: `http://inference:8000/metrics`.

- Observability & keys to avoid renaming:
	- Structured log keys used by dashboards: `skip_idempotent`, `train_success_publish`, `promotion_scoreboard`, `promotion_artifacts_ok`, `queue_enqueued`, `predict_inference_start`.

- Common pitfalls (callouts):
	- Missing scaler in MLflow artifacts → inference load errors.
	- Mismatched bucket names between training and inference → silent upload failures.
	- Promotion pointer JSON schema changes break pointer resolution; paths: `model-promotion/<identifier|global>/<config_hash>/promotion-*.json` and `current.json`.

- Quick heuristics for code edits:
	- When adding a trainer, update `EXPECTED_MODEL_TYPES`, add a compose service with a unique `CONSUMER_GROUP_ID`, and ensure eval promotion pointers remain resolvable.
	- When changing the config-hash logic, update parquet metadata writing in `preprocess_container/main.py` and tests that expect idempotency.

Update this file if bucket names, promotion schemas, or backpressure conventions change. For other details, see `README.md`, `BACKPRESSURE_NOTES.md`, and container `main.py` files referenced above.
