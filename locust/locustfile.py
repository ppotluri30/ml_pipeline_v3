"""
Focused Locust load test for inference performance.

Goal:
    - Drive /predict (inference) with 80% task weight using POST /predict and minimal body {"inference_length": 1}.
    - Keep /healthz (10%) and a /download probe (10%) as background noise.
    - Record ALL requests (success + failure) with latency to JSONL for offline analysis.

Usage (from UI after `docker compose up -d locust`):
    Users: 25
    Spawn rate: 5
    Host: http://inference:8000
    Start swarming – watch /predict median, p95, p99.

CSV Export: If you also want CSVs, launch Locust with `--csv /mnt/locust/results --csv-full-history` (compose command not modified here).
"""

from locust import HttpUser, task, between, events
import os, json, time, threading, uuid

# NOTE: We intentionally removed earlier endpoint discovery complexity.
# The host will be provided via the Locust UI as http://inference:8000.
# Tasks use relative paths so they follow the configured host.

# --- Configuration via environment variables ---
EP_HEALTH = os.getenv("ENDPOINT_HEALTH", "/healthz")
# Download endpoint must target the gateway (fastapi-app), not the inference container.
# Default to the test parquet (smaller) but allow override.
GATEWAY_BASE = os.getenv("GATEWAY_BASE", "http://fastapi-app:8000")
EP_DOWNLOAD = os.getenv("ENDPOINT_DOWNLOAD", f"{GATEWAY_BASE}/download/processed-data/test_processed_data.parquet")
EP_DOWNLOAD_ALT = os.getenv("ENDPOINT_DOWNLOAD_ALT", f"{GATEWAY_BASE}/download/processed-data/processed_data.parquet")
# Force absolute predict URL (remove ambiguity about host)
PREDICT_URL = os.getenv("PREDICT_URL", "http://inference:8000/predict")
LOG_FILE = os.getenv("LOG_FILE", "/mnt/locust/locust_requests.jsonl")
TRUNCATE_ON_START = os.getenv("LOCUST_TRUNCATE_LOG", "0") == "1"
DOWNLOAD_WARMUP_ATTEMPTS = int(os.getenv("DOWNLOAD_WARMUP_ATTEMPTS", "5"))
DOWNLOAD_WARMUP_DELAY_SEC = float(os.getenv("DOWNLOAD_WARMUP_DELAY", "0.5"))
_predict_ready = True  # warmup removed; assume service pre-warmed
_download_ready = False
_download_warm_attempts = 0
_download_active_url: str | None = None
_headless_auto_started = False

# Normalize in case someone passed relative paths via env
if EP_DOWNLOAD.startswith('/'):
    EP_DOWNLOAD = f"{GATEWAY_BASE.rstrip('/')}{EP_DOWNLOAD}"
if EP_DOWNLOAD_ALT and EP_DOWNLOAD_ALT.startswith('/'):
    EP_DOWNLOAD_ALT = f"{GATEWAY_BASE.rstrip('/')}{EP_DOWNLOAD_ALT}"

def _is_headless():
    # Locust sets environment.parsed_options.headless when running headless
    try:
        from locust.env import Environment  # type: ignore
        return bool(events.request._handlers and any(True for _ in [1]))  # crude presence check
    except Exception:
        return False

@events.test_start.add_listener
def on_test_start(environment, **kw):  # noqa: D401
    # In headless mode, ensure at least one user executes warm-up immediately.
    global _headless_auto_started
    if _headless_auto_started:
        return
    _headless_auto_started = True
    # Nothing else required; the first spawned users will trigger warm-up in on_start.
    _append_jsonl({
        "ts": time.time(),
        "event": "headless_test_start",
    })

_log_file_lock = threading.Lock()

# --- Session / run identification ---
# A unique run identifier to delineate test sessions in the JSONL log.
RUN_ID = os.getenv("LOCUST_RUN_ID") or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
_session_header_written = False

# No discovery – assume inference service already promoted & loaded.

def _append_jsonl(record: dict):
    """Thread-safe append of a single JSON record to the log file."""
    try:
        line = json.dumps(record, separators=(",", ":"))
        with _log_file_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as e:
        # Avoid throwing inside hook; just print.
        print(f"[locustfile] Failed logging record: {e}")


@events.request.add_listener
def log_request(request_type, name, response_time, response_length, response, context, exception, **kw):  # noqa: D401
    """Selective logging:
    - Suppress predict noise until warm-up succeeds.
    - After ready, log only successful (200) predict requests.
    - Always log health/download.
    """
    try:
        status_code = getattr(response, "status_code", None) if response else None
        if name == "predict":
            if not _predict_ready:
                return
            if status_code != 200:
                return
        _append_jsonl({
            "ts": time.time(),
            "request_type": request_type,
            "name": name,
            "response_time_ms": response_time,
            "status_code": status_code,
            "error": str(exception) if exception else None,
        })
    except Exception as e:  # pragma: no cover
        print(f"[locustfile] log_request hook failed: {e}")


class PipelineUser(HttpUser):
    """User model focused on inference performance (predict 80%, health 10%, download 10%)."""
    wait_time = between(1, 2)


    def _download_warmup(self):
        global _download_ready, _download_warm_attempts
        if _download_ready:
            return
        global _download_active_url
        for attempt in range(1, DOWNLOAD_WARMUP_ATTEMPTS + 1):
            _download_warm_attempts = attempt
            ok = False
            chosen = None
            try:
                r = self.client.get(EP_DOWNLOAD, name="download_warm", timeout=10)
                if r.status_code == 200:
                    ok = True
                    chosen = EP_DOWNLOAD
                elif EP_DOWNLOAD_ALT:
                    r2 = self.client.get(EP_DOWNLOAD_ALT, name="download_warm_alt", timeout=10)
                    if r2.status_code == 200:
                        ok = True
                        chosen = EP_DOWNLOAD_ALT
            except Exception:
                ok = False
            if ok:
                _download_ready = True
                if chosen:
                    _download_active_url = chosen
                _append_jsonl({
                    "ts": time.time(),
                    "event": "download_warmup_success",
                    "attempt": attempt
                })
                return
            else:
                _append_jsonl({
                    "ts": time.time(),
                    "event": "download_warmup_attempt",
                    "attempt": attempt
                })
                time.sleep(DOWNLOAD_WARMUP_DELAY_SEC)
        _append_jsonl({
            "ts": time.time(),
            "event": "download_warmup_failed",
            "attempts": DOWNLOAD_WARMUP_ATTEMPTS
        })

    def on_start(self):
        global _session_header_written
        if not _session_header_written:
            # Optionally truncate log file to avoid historical noise (legacy 'inference-discovery' entries, etc.)
            if TRUNCATE_ON_START:
                try:
                    open(LOG_FILE, "w").close()
                except Exception:
                    pass
            _append_jsonl({
                "ts": time.time(),
                "event": "session_start",
                "run_id": RUN_ID,
            })
            _session_header_written = True
        _append_jsonl({
            "ts": time.time(),
            "event": "config_snapshot",
            "run_id": RUN_ID,
            "ep_health": EP_HEALTH,
            "ep_download": EP_DOWNLOAD,
            "ep_download_alt": EP_DOWNLOAD_ALT,
            "gateway_base": GATEWAY_BASE,
            "predict_url": PREDICT_URL,
            "predict_warmup_removed": True,
            "download_warmup_attempts": DOWNLOAD_WARMUP_ATTEMPTS,
        })
        # Warm health once (ignored errors)
        try:
            self.client.get(EP_HEALTH, name="healthz-warm")
        except Exception:
            pass
    # Predict warmup removed; direct traffic starts immediately.
        self._download_warmup()

    @task(80)
    def predict(self):
        if not _predict_ready:
            return
        self.client.post(PREDICT_URL, json={"inference_length": 1}, name="predict")

    @task(10)
    def health(self):
        self.client.get(EP_HEALTH, name="healthz")

    @task(10)
    def download_processed(self):
        if not _download_ready:
            return
        # Use the active URL determined during warmup; if missing fallback to primary.
        url = _download_active_url or EP_DOWNLOAD
        self.client.get(url, name="download_processed")


__all__ = ["PipelineUser"]
if __name__ == "__main__":
    # Allow running directly for quick local debug
    from locust import run_single_user
    run_single_user(PipelineUser)
