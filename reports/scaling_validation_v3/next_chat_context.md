## Next-agent onboarding context

Date: 2025-10-28

Purpose
- Provide a concise, copy-pasteable context so the next agent can pick up exactly where work left off: canonical rerun artifacts, aggregated reports, pending validation steps, and recommended next commands.

Summary of work completed
- Performed and promoted a clean 8-replica rerun (canonical 8×400) and archived the previous failing run under an `_obsolete` folder.
- Rebuilt scaling metrics/metadata and regenerated visuals and a summary report (Markdown + PDF) in `reports/scaling_validation_v3`.
- Captured CPU snapshots and logs; confirmed zero failures for the canonical 8×400 run.

Primary artifact locations
- Canonical locust results: `locust/results/multi_replica_v1/replicas8_u400`
- Archived failing run: `locust/results/multi_replica_v1/replicas8_u400_obsolete`
- Aggregated metrics & metadata: `reports/scaling_validation_v3/scaling_metrics.csv`, `reports/scaling_validation_v3/scaling_metrics.json`, `reports/scaling_validation_v3/scaling_metadata.csv`, `reports/scaling_validation_v3/scaling_metadata.json`
- Visuals & narrative: `reports/scaling_validation_v3/*.png`, `reports/scaling_validation_v3/scaling_report.md`, `reports/scaling_validation_v3/scaling_report.pdf`
- CPU snapshots & logs: see `locust/results/.../*_cpu_snapshot*.txt` and `inference_logs.txt` in repo root.

Key runtime facts & observations
- Docker Compose stack manifest is at repository root: `docker-compose.yaml` (contains kafka, minio, postgres, mlflow, fastapi-app, preprocess, train_gru/train_lstm/nonml_prophet, eval, inference, inference-lb, locust master/worker).
- The `preprocess` service was originally run with sampling env vars (`SAMPLE_TRAIN_ROWS`, `SAMPLE_TEST_ROWS`, `SAMPLE_STRATEGY`) which caused partial ingestion; those were removed/overridden to run a full-dataset ingestion for the canonical rerun.
- Promoted model layout follows MLflow artifact conventions (model folder named by `MODEL_TYPE` and containing `scaler/*.pkl`); missing scaler will break inference.
- Canonical 8×400 run metrics: ~56 req/s, 0 failures; CPU evidence shows container-level saturation (~120–150% CPU per container) indicating compute-bound inference.

Pending tasks / recommended next actions
1. Decide whether to run the remaining validation: full sweep of replicas (1 → 2 → 4 → 8) against user tiers (200, 400, 800). This will map saturation points fully.
2. If yes, prepare to run the full-dataset training in this environment (ensure sufficient CPU/RAM and disk for MLflow/MinIO/Postgres) or reuse the existing trained artifacts.
3. If automation is desired, create the helper scripts: `run_full_pipeline.ps1` (PowerShell) and `run_scaling_sweep.ps1` to perform compose up/down, scale inference replicas, and run Locust headless runs while collecting results.

Quick runbook (copyable commands for the next agent)

1) Remove sampling so preprocess ingests full dataset (edit `docker-compose.yaml` to unset/remove these env lines in the `preprocess` service):

```powershell
# Edit file in your editor or use a script to remove/unset these env vars:
# SAMPLE_TRAIN_ROWS=51
# SAMPLE_TEST_ROWS=31
# SAMPLE_STRATEGY=head
# Optionally set FORCE_REPROCESS=1 to force rebuild
```

2) Start core infra and services (PowerShell; run from repo root):

```powershell
& .\.venv\Scripts\Activate.ps1
docker compose up -d kafka minio postgres mlflow fastapi-app minio-init
# Wait until MLflow/minio healthy then start pipeline services
docker compose up -d preprocess train_gru train_lstm nonml_prophet eval inference inference-lb
# Tail logs while watching for promotions
docker compose logs -f eval inference fastapi-app
```

3) Watch for model promotion (eval will write promotion JSON under MinIO/`model-promotion` and FastAPI will expose it at `/download/model-promotion/current.json`). Check inference readiness on the load balancer:

```powershell
Invoke-RestMethod -Uri http://localhost:8023/metrics | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8023/healthz
# Or check FastAPI promotion file
Invoke-RestMethod -Uri http://localhost:8000/download/model-promotion/current.json | ConvertTo-Json
```

4) Run the Locust swarm headless (example for 200 users, 2m):

```powershell
# Start master (if using compose locust service) and scale workers
docker compose up -d locust
docker compose up -d --scale locust-worker=4 locust-worker
# Run a headless master directly if preferred:
docker compose run --rm --service-ports locust /bin/sh -c "locust -f /mnt/locust/locustfile.py --headless -u 200 -r 20 -t 2m --host http://inference-lb"
```

5) Scale inference service and repeat tests (example):

```powershell
docker compose up -d --scale inference=2
# run locust tiers and collect results
docker compose up -d --scale inference=4
# run locust tiers
docker compose up -d --scale inference=8
# run locust tiers
```

6) Collect CPU snapshots and metrics for each run; put them under `locust/results/multi_replica_v1/<label>` and then run the aggregation scripts to regenerate `reports/scaling_validation_v3/*`.

Decision checklist for next agent
- Confirm whether to run full sweep across replicas/users or only selected tiers.
- Confirm whether to run full-dataset training in this environment (resource-heavy) or reuse existing trained model artifacts.
- If running locally, ensure the host has enough CPU/RAM or use a cloud instance for load generation (especially for 800-user tier).

Contact points in repo
- Aggregation & plotting scripts already used during previous run exist in the repo root and `locust` folder (search for `scaling` or `aggregate` scripts if needed).

Success criteria for the sweep
- A model is trained, evaluated, and promoted (eval promotion JSON visible via FastAPI endpoint).
- For each inference replica scale (1/2/4/8) and user tiers (200/400/800), we have locust result directories with throughput, latencies, and failure counts; CPU snapshots captured.
- A consolidated `scaling_metrics.csv`, charts (`.png`), and a final Markdown+PDF report created under `reports/scaling_validation_v3/`.

Notes
- Full-dataset training is resource- and time-intensive. If time/CPU is limited, consider running only inference + locust sweep using produced artifacts.
- If MLflow trainer fails to produce `scaler/*.pkl`, inference will fail. Verify trainer logs and artifacts before running heavy load.

---

If you want, I can also add automation scripts (PowerShell) and the aggregation notebook into this repo now — say `yes` and I will create `run_full_pipeline.ps1`, `run_scaling_sweep.ps1`, and an `aggregation/` script and test them locally where possible.
