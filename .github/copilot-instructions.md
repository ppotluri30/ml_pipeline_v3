## FLTS AI Agent Guide (Concise, Actionable)
## FLTS AI Agent Guide — concise, actionable (~30 lines)

This file captures the must-know repo-level patterns for an automated coding agent working on the FLTS pipeline.

- Big picture: claim-check pipeline: preprocess -> train -> eval/promotion -> inference. Communication is via Kafka claim JSONs and object storage accessed through the FastAPI gateway endpoints in the repo (see `inference_container/api_server.py` and gateway callers across containers).

- Key code locations (read these first):
	- `preprocess_container/main.py` (build_active_config, config_hash, parquet metadata)
	- `train_container/main.py` (extract meta, windowing, artifact layout)
	- `eval_container/main.py` (promotion logic, scoring, pointer files)
	- `inference_container/inferencer.py` + `inference_container/main.py` (model loading order, prediction dedupe)
	- `locust/locustfile.py` (load-testing scenarios)

- Canonical config & lineage: preprocess builds a canonical JSON (sorted compact) + `_data` + `EXTRA_HASH_SALT` → SHA256 `config_hash`. The hash is stored in parquet metadata (`preprocess_config`, `config_hash`) and used to deduplicate work across services.

- Claim JSON shape (examples you can rely on):
	- Preprocess -> {bucket:"processed-data", object:"processed_data.parquet", config_hash:"<sha256>", identifier:"id"}
	- Train SUCCESS -> {operation:"Trained: <MODEL_TYPE>", status:"SUCCESS", config_hash:"<sha256>", run_id:"<mlflow_run>"}
	- Promotion -> {model_uri:"runs:/<run_id>/<MODEL_TYPE>", config_hash:"<sha256>", ...}

- Important envs and conventions (used widely across containers):
	- EXTRA_HASH_SALT, FORCE_REPROCESS, SAMPLE_TRAIN_ROWS, SAMPLE_TEST_ROWS
	- SKIP_DUPLICATE_CONFIGS, DUP_CACHE_MAX (trainer dedup cache)
	- EXPECTED_MODEL_TYPES (eval waits for these), PROMOTION_SEARCH_RETRIES/DELAY
	- INFERENCE_* gates: WAIT_FOR_MODEL, MODEL_WAIT_TIMEOUT, INFERENCE_PREWARM, RUN_INFERENCE_ON_TRAIN_SUCCESS, QUEUE_WORKERS, QUEUE_MAXSIZE

- Artifacts & buckets: trainers write model artifacts under a folder named exactly `MODEL_TYPE`; scaler(s) under `scaler/*.pkl`. Buckets are created by `_ensure_buckets()` (train/eval/inference). Key bucket names: `processed-data`, `mlflow`, `model-promotion`, `inference-logs`. Watch out: trainer default `INFERENCE_LOG_BUCKET` sometimes differs (`inference-txt-logs`) — set envs consistently.

- Inference startup order and loading: pointers resolved in order `current.json` (root) → `global/current.json` → `<identifier>/current.json`. Model URIs try `runs:/<run_id>/<run_name>` then `runs:/<run_id>/model` fallback. Prediction dedupe uses `(run_id, prediction_hash)`.

- Scoring & promotion: eval implements a weighted score (0.5*rmse + 0.3*mae + 0.2*mse), lower is better; ties resolved by newest start_time. Promotion writes history under `model-promotion/<id|global>/<config_hash>/promotion-<ts>.json` and updates `current.json` pointers before publishing `model-selected` claims.

- Common gotchas (do not fix blindly):
	- Publishing SUCCESS before artifacts exist → eval can't load model.
	- Dropping/renaming sin/cos time features (min_of_day, day_of_week, day_of_year) → inference shape failures.
	- Forgetting to drop original target after synthesizing `value` in trainer → feature-count mismatch.
	- Missing bucket in `_ensure_buckets()` → silent artifact/log loss.

- Local dev & debug commands (fast path):
	- Full dev loop: set SAMPLE_* envs and EXTRA_HASH_SALT, then:
		docker compose up --build preprocess train_gru train_lstm nonml_prophet eval inference
	- Run only inference: docker compose up --build inference
	- Locust load test: run the `locust` service and open http://localhost:8089 (see `locust/locustfile.py`).

- How to extend: to add a new MODEL_TYPE add a branch in trainer `_build_model`, ensure artifact folder == MODEL_TYPE, add a `train_<name>` service in compose with unique `CONSUMER_GROUP_ID`, and add it to `EXPECTED_MODEL_TYPES` in eval.

- Logging: services write one-line JSON logs with stable keys (examples: `skip_idempotent`, `target_fallback`, `feature_count_mismatch`, `promotion_waiting_for_models`, `promotion_scoreboard`, `promotion_artifacts_ok`, `predict_inference_start`). Do not rename these keys — dashboards and parsers rely on them.

- Where to look for tests & quick checks: `inference_container/tests/` and `locust/` — prefer running lighter local smoke runs before full compose.

Update this file when bucket names, pointer formats, env conventions, or model families change. Keep it short and concrete.
