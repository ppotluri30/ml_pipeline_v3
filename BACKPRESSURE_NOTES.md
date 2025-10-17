# Backpressure & Load Validation Snapshot

## Session snapshot — 2025-10-16

- **Inference fixes now deployed**
	- Rebuilt container after updating `inference_container/process_pool.py` so `InferenceHTTPError` is picklable and the executor always logs `process_pool_context_selected=spawn`.
	- Disabled FastAPI predict cache via `ENABLE_PREDICT_CACHE=0` in `docker-compose.yaml`; service env still exports `DISABLE_INFERENCE_CACHE=1` for parity.
	- Confirmed model pointer (`runs:/0f36e085d77441aca95e0d722f655d95`) loads with scaler auto-discovery; no further config tweaks required.
- **Validation status**
	- `payload-valid.json` (30 rows; satisfies `input_seq_len + output_seq_len`) → `POST /predict` returns 200 with predictions captured in `response.json`.
	- `payload-invalid.json` (missing required columns) → `POST /predict` returns expected 400; body stored in `response-invalid.json`.
	- Process pool remains healthy: log tail shows only `queue_job_enqueued/start/done`; no `queue_job_error` nor `BrokenProcessPool` after multiple requests.
- **Testing progress**
	- Health checks complete; ready to grab Prometheus snapshots (`inference_workers_busy`, queue depth, CPU) and launch headless Locust smoke next.
	- Pending follow-up: run headless Locust scenario (reuse prior profile: 80 users, spawn 8, 60s) and capture metrics/summary; verify `/metrics` scrape post-run.
- **Artifacts & references**
	- Validation payloads: `payload-valid.json`, `payload-invalid.json`.
	- Latest responses: `response.json`, `response-invalid.json`.
	- Logs already inspected via `docker compose logs inference --tail 120` (clean) and metrics query `max_over_time(inference_workers_busy[30s])` (pending inclusion in summary once gathered).

This section records the active state so the next session can continue with metrics capture and load validation without repeating setup.

This note captures the validated staging configuration, a metrics snapshot for the inference service, and results from a short headless load test. Use it as a reference for future runs and tuning.

## Staging configuration (persisted)

- Inference service (env flags)
	- USE_BOUNDED_QUEUE=true
	- USE_MANUAL_COMMIT=true
	- ENABLE_MICROBATCH=true
	- ENABLE_TTL=true
	- ENABLE_PUBLISH_API=true
	- UVICORN_KEEPALIVE=30
	- UVICORN_TIMEOUT_KEEP_ALIVE=60
	- UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN=30
	- QUEUE_MAXSIZE=40
	- PAUSE_THRESHOLD_PCT=80
	- RESUME_THRESHOLD_PCT=50
- Locust master (headless)
	- LOCUST_HEADLESS=1
	- Default persisted: USERS=200, SPAWN_RATE=20, RUNTIME=120s
	- Kafka burst seeding enabled: KAFKA_BURST=1, KAFKA_BURST_COUNT=1
	- Predict warm-up disabled: PREDICT_WARMUP_DISABLE=1
- Metrics sidecar (alpine + docker-cli)
	- Prints docker stats for the flts-main-inference-1 container every ~5s
	- /var/run/docker.sock mounted read-only

Note: IDENTIFIER was not set; Docker logged warnings. Optional: set IDENTIFIER in staging compose to remove noise.

## Metrics sidecar snapshot (inference)

- Example readings while idle/light load:
	- CPU: 14–27% (spikes observed initially up to ~195% during startup/activity)
	- Memory: ~585–590 MiB (stable after warm)
	- PIDs: ~140–160
	- NET/BLOCK I/O: low under light load; expected to rise with active tests

These values are printed periodically by the metrics sidecar and can be tailed via compose logs.

## Headless smoke results (validated)

- Run: 20 users, spawn rate 10, 45s; Kafka burst seeding enabled
- Outcome: 0 failures across 192 requests
	- POST /predict: 138 calls, 0 failures, median ~4ms, p95 ~7–10ms
	- /healthz and data downloads: 0 failures
- Inference logs showed predict_served_cached_direct, confirming cached fast-path was primed after Kafka burst + readiness polling.

## Notes and tips

- Cached fast-path: Keep Kafka burst seeding and readiness polling in place to ensure has_df=true before ramping load; reduces initial 429/RemoteDisconnected risks.
- Resource telemetry: Enable ENABLE_RESOURCE_LOGS=1 on inference to emit structured CPU/MEM logs from the app (psutil-based) if you prefer in-process telemetry alongside the sidecar.
- Scaling up: For larger runs (e.g., 200 users / 120s), reuse the same envs. Watch docker stats and adjust QUEUE_MAXSIZE, pause/resume thresholds, or Locust spawn rates as needed.

## Quick checklist for future runs

- [ ] Compose up with staging overrides; ensure metrics sidecar is started
- [ ] Trigger Kafka burst seeding; wait for /predict_ping to report ready (handled by Locust harness)
- [ ] Run headless test; confirm 0% failures before increasing scale
- [ ] Capture CPU/MEM snapshots and any anomalies (I/O spikes, queue depth logs)

---
This document reflects the state validated on the last run and should be updated if flags, thresholds, or model families change.

## Locust Web UI (manual testing) and headless (CI)

- Web UI is enabled by default via the Locust service command in `docker-compose.yaml`.
- Access: http://localhost:8089
- To run headless instead (e.g., CI), set `LOCUST_MODE=headless` and optionally `USERS`, `SPAWN_RATE`, and `RUNTIME`.
- Staging overrides (`docker-compose.staging.yaml`) can set `LOCUST_MODE=headless` for automated validations while keeping UI available when toggled.

Windows/PowerShell gotcha and fix:
- Multi-line shell conditionals in Compose commands are fragile on Windows due to CRLF and quoting semantics. Symptoms: container exits immediately; logs show `[: 1: Syntax error: end of file unexpected (expecting "then")` or no output.
- Fix: avoid `/bin/sh -c` conditionals in Compose; instead, use the Locust image default entrypoint and an explicit command array for UI mode:
	- command: ["-f","/mnt/locust/locustfile.py","--host","http://inference:8000","--web-host","0.0.0.0","--web-port","8089"]
- Headless alternative (if needed): create a separate override profile or service using command array
	- command: ["-f","/mnt/locust/locustfile.py","--headless","-u","200","-r","20","--run-time","120s","--host","http://inference:8000"]

Verification checklist:
- `docker compose ps locust` shows `Up` with `0.0.0.0:8089->8089/tcp` mapping.
- `docker logs flts-main-locust-1` includes: `Starting web interface at http://0.0.0.0:8089`.
- `curl http://localhost:8089` returns HTTP 200 from the host. If it returns 200 inside the container but not on host, check firewall/VPN.

## Warm-up stability

Observed 500s root cause:
- The `/predict_warmup` requests happened before the inference service had a cached dataframe/model ready (predict_ping showed busy/has_df false), causing 500 Server Error responses from the `/predict` path used by warm-up.

Safe resolution applied (non-disruptive):
- Disabled Locust predict warm-up via `PREDICT_WARMUP_DISABLE=1` in staging overrides.
- Enabled a minimal Kafka claim burst to prime readiness: `KAFKA_BURST=1`, `KAFKA_BURST_COUNT=1`. The Locust harness waits on `predict_ping` indicating `has_df=true` before traffic ramps.

How to toggle:
- Enable warm-up: set `PREDICT_WARMUP_DISABLE=0` (or unset) and ensure readiness delay/retries are sufficient.
- Disable warm-up: set `PREDICT_WARMUP_DISABLE=1` to skip client-side seeding; rely on claim-check and readiness polling.

Verification steps:
- Rebuild/restart Locust, run a 30s test (UI or headless); check Locust report and `locust_requests.jsonl` for zero 500s under `predict_warmup`.
