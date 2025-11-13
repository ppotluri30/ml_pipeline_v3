"""
Focused Locust load test for inference performance.

Goal:
    - Drive /predict (inference) with 80% task weight using POST /predict and minimal body {"inference_length": 1}.
    - Keep /healthz (10%) and a /download probe (10%) as background noise.
    - Record ALL requests (success + failure) with latency to JSONL for offline analysis.

Usage (from UI after `docker compose up -d locust`):
    Users: 25
    Spawn rate: 5
    Host: http://inference-lb
    Start swarming – watch /predict median, p95, p99.

CSV Export: If you also want CSVs, launch Locust with `--csv /mnt/locust/results --csv-full-history` (compose command not modified here).
"""

from locust import HttpUser, task, between, events
import os, json, time, threading, uuid, random, datetime as dt
import posixpath
from urllib.parse import urlsplit

# NOTE: We intentionally removed earlier endpoint discovery complexity.
# The host will be provided via the Locust UI as http://inference-lb.
# Tasks use relative paths so they follow the configured host.

# --- Configuration via environment variables ---
EP_HEALTH = os.getenv("ENDPOINT_HEALTH", "/healthz")
# Download endpoint must target the gateway (fastapi-app), not the inference container.
# Default to the test parquet (smaller) but allow override.
GATEWAY_BASE = os.getenv("GATEWAY_BASE", "http://fastapi-app:8000")
EP_DOWNLOAD = os.getenv("ENDPOINT_DOWNLOAD", f"{GATEWAY_BASE}/download/processed-data/test_processed_data.parquet")
EP_DOWNLOAD_ALT = os.getenv("ENDPOINT_DOWNLOAD_ALT", f"{GATEWAY_BASE}/download/processed-data/processed_data.parquet")
# Force absolute predict URL (remove ambiguity about host)
RAW_PREDICT_URL = os.getenv("PREDICT_URL", "http://inference-lb/predict").strip() or "/predict"
_target_host_env = (os.getenv("TARGET_HOST") or os.getenv("LOCUST_DEFAULT_HOST") or "http://inference-lb").strip()

if "://" not in RAW_PREDICT_URL:
    if "://" not in _target_host_env:
        _target_host_env = f"http://{_target_host_env.lstrip('/')}"
    _parsed_host = urlsplit(_target_host_env)
    if not _parsed_host.scheme or not _parsed_host.netloc:
        raise RuntimeError(f"Invalid TARGET_HOST/LOCUST_DEFAULT_HOST value '{_target_host_env}'")
    _predict_host = f"{_parsed_host.scheme}://{_parsed_host.netloc}"
    _predict_path = RAW_PREDICT_URL if RAW_PREDICT_URL.startswith("/") else f"/{RAW_PREDICT_URL}"
    PREDICT_URL = f"{_predict_host}{_predict_path}"
else:
    _parsed_url = urlsplit(RAW_PREDICT_URL)
    if not _parsed_url.scheme or not _parsed_url.netloc:
        raise RuntimeError(f"Invalid PREDICT_URL '{RAW_PREDICT_URL}'")
    _predict_host = f"{_parsed_url.scheme}://{_parsed_url.netloc}"
    _predict_path = _parsed_url.path or "/predict"
    if _parsed_url.query:
        _predict_path = f"{_predict_path}?{_parsed_url.query}"
    PREDICT_URL = RAW_PREDICT_URL

_predict_path_only = _predict_path.split("?", 1)[0] or "/"
_predict_parent_path = _predict_path_only.rsplit("/", 1)[0] or "/"
_predict_parent_path = posixpath.normpath(_predict_parent_path)
if not _predict_parent_path.startswith("/"):
    _predict_parent_path = f"/{_predict_parent_path}"
if _predict_parent_path == ".":
    _predict_parent_path = "/"
_predict_parent_url = _predict_host if _predict_parent_path in {"", "/"} else f"{_predict_host}{_predict_parent_path}"
_predict_request_path = _predict_path


def _predict_api_url(segment: str) -> str:
    segment_clean = segment.lstrip("/")
    base = _predict_parent_url.rstrip("/")
    if not base:
        base = _predict_host.rstrip("/")
    return f"{base}/{segment_clean}" if segment_clean else base


def _predict_api_path(segment: str) -> str:
    parent = _predict_parent_path if _predict_parent_path else "/"
    combined = posixpath.join(parent, segment.lstrip("/"))
    combined = posixpath.normpath(combined)
    if not combined.startswith("/"):
        combined = f"/{combined}"
    return combined


_predict_ping_url = _predict_api_url("predict_ping")
_predict_ping_path = _predict_api_path("predict_ping")
_predict_publish_url = _predict_api_url("publish_inference_claims")
LOG_FILE = os.getenv("LOG_FILE", "/mnt/locust/locust_requests.jsonl")
TRUNCATE_ON_START = os.getenv("LOCUST_TRUNCATE_LOG", "0") == "1"
DOWNLOAD_WARMUP_ATTEMPTS = int(os.getenv("DOWNLOAD_WARMUP_ATTEMPTS", "5"))
DOWNLOAD_WARMUP_DELAY_SEC = float(os.getenv("DOWNLOAD_WARMUP_DELAY", "0.5"))
# Enhanced predict logging controls
LOG_PREDICT_ALL = os.getenv("LOG_PREDICT_ALL", "0") == "1"  # log every predict (success + failure)
LOG_PREDICT_ERRORS = os.getenv("LOG_PREDICT_ERRORS", "1") == "1"  # log failed predict even if not logging all
LOG_PREDICT_RESPONSE_CHARS = int(os.getenv("LOG_PREDICT_RESPONSE_CHARS", "0"))  # capture first N chars of body
LOG_PREDICT_PAYLOAD = os.getenv("LOG_PREDICT_PAYLOAD", "0") == "1"  # echo JSON payload (small tests only)
PREDICT_WARMUP_DISABLE = os.getenv("PREDICT_WARMUP_DISABLE", "0") in {"1", "true", "TRUE"}
_predict_cache_flag = os.getenv("LOCUST_ENABLE_PREDICT_CACHE", os.getenv("ENABLE_PREDICT_CACHE", "0"))
_predict_cache_enabled = str(_predict_cache_flag).lower() in {"1", "true", "yes"}
_predict_ready = False  # require successful warm-up predict before sending cached requests
_warmup_done = False
_warmup_lock = threading.Lock()
_download_ready = False
_download_warm_attempts = 0
_download_active_url: str | None = None
_headless_auto_started = False

# Predict payload handling. Auto mode will fall back to synthetic payloads when the
# inference service reports no cached dataframe.
PREDICT_PAYLOAD_MODE = os.getenv("PREDICT_PAYLOAD_MODE", "auto").strip().lower()
_predict_input_len = 10
_predict_output_len = 1
_predict_has_df = False
_predict_payload_seq = 0
_predict_payload_lock = threading.Lock()

# Allow tuning of user pacing without editing the file.
try:
    _user_wait_min = float(os.getenv("PREDICT_USER_WAIT_MIN", "1"))
except Exception:
    _user_wait_min = 1.0
try:
    _user_wait_max = float(os.getenv("PREDICT_USER_WAIT_MAX", "2"))
except Exception:
    _user_wait_max = 2.0
if _user_wait_max < _user_wait_min:
    _user_wait_max = _user_wait_min

# Optional: trigger a Kafka burst via inference API's /publish_inference_claims
KAFKA_BURST = os.getenv("KAFKA_BURST", "0") in {"1", "true", "TRUE"}
KAFKA_BURST_COUNT = int(os.getenv("KAFKA_BURST_COUNT", "0"))
KAFKA_BURST_TTL_MS = os.getenv("KAFKA_BURST_TTL_MS")
KAFKA_BURST_KEY_PREFIX = os.getenv("KAFKA_BURST_KEY_PREFIX")

# Normalize in case someone passed relative paths via env
if EP_DOWNLOAD.startswith('/'):
    EP_DOWNLOAD = f"{GATEWAY_BASE.rstrip('/')}{EP_DOWNLOAD}"
if EP_DOWNLOAD_ALT and EP_DOWNLOAD_ALT.startswith('/'):
    EP_DOWNLOAD_ALT = f"{GATEWAY_BASE.rstrip('/')}{EP_DOWNLOAD_ALT}"


def _resolve_predict_lengths(timeout: float = 10.0) -> tuple[int, int, bool]:
    """Query /predict_ping to discover model sequence lengths.

    Returns (input_seq_len, output_seq_len, has_df_cached).
    """
    ping_url = _predict_ping_url
    in_len = 10
    out_len = 1
    has_df = False
    try:
        import requests  # local import to avoid hard dependency in unit context
        resp = requests.get(ping_url, timeout=timeout)
        if resp is not None and resp.status_code == 200:
            try:
                payload = resp.json()
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                in_len = int(payload.get("input_seq_len") or in_len)
                out_len = int(payload.get("output_seq_len") or out_len)
                has_df = bool(payload.get("has_df"))
    except Exception:
        pass
    return in_len, out_len, has_df


def _build_synthetic_predict_payload(
    input_len: int,
    output_len: int,
    total_rows: int | None = None,
    base_time: dt.datetime | None = None,
    freq_minutes: int = 1,
) -> dict:
    """Construct a schema-aligned synthetic payload for /predict.

    The column order mirrors the numeric network metrics used during training.
    CRITICAL: Ensures unique, monotonically increasing timestamps to avoid
    'Time frequency is zero' errors during high-concurrency inference.
    """
    rows_needed = max(input_len + max(output_len, 1) + 5, 16)
    total = total_rows if total_rows is not None else rows_needed
    if base_time is None:
        t0 = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    else:
        t0 = base_time
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=dt.timezone.utc)
        t0 = t0.astimezone(dt.timezone.utc).replace(microsecond=0)
    
    # CRITICAL FIX: Ensure freq_minutes is always >= 1 to guarantee unique timestamps
    step = max(1, int(freq_minutes))
    
    # Generate unique timestamps with guaranteed spacing
    times_dt = [t0 + dt.timedelta(minutes=i * step) for i in range(total)]
    
    # CRITICAL: Use explicit format to ensure pandas can parse correctly
    # Format: "YYYY-MM-DDTHH:MM:SS" without microseconds or timezone
    times = [ts.strftime("%Y-%m-%dT%H:%M:%S") for ts in times_dt]
    
    # Verify uniqueness (debug assertion)
    if len(times) != len(set(times)):
        raise ValueError(f"Generated {len(times)} timestamps but only {len(set(times))} are unique!")
    
    # ===== DIAGNOSTIC LOGGING =====
    DEBUG_ENABLED = os.getenv("DEBUG_LOCUST_PAYLOAD", "0") in {"1", "true", "TRUE"}
    if DEBUG_ENABLED:
        print(f"[LOCUST_GENERATE] Generated {len(times)} timestamps")
        print(f"[LOCUST_GENERATE] Unique count: {len(set(times))}")
        print(f"[LOCUST_GENERATE] Sample: {times[:3]}")
        print(f"[LOCUST_GENERATE] Field name will be: 'ts'")
    # ===== END DIAGNOSTIC LOGGING =====
    
    base_seq = [float(i % 50) for i in range(total)]
    down_series = [v * 1_000_000.0 + 5_000_000.0 for v in base_seq]
    data = {
        "ts": times,
        "down": down_series,
        "up": [v * 1000.0 + 1000.0 for v in base_seq],
        "rnti_count": [2000.0 + v for v in base_seq],
        "mcs_down": [10.0 + (v % 5) for v in base_seq],
        "mcs_down_var": [50.0 + (v * 0.5) for v in base_seq],
        "mcs_up": [12.0 + (v % 4) for v in base_seq],
        "mcs_up_var": [40.0 + (v * 0.4) for v in base_seq],
        "rb_down": [0.05 + (v * 0.001) for v in base_seq],
        "rb_down_var": [1e-7 + (v * 1e-9) for v in base_seq],
        "rb_up": [0.01 + (v * 0.0005) for v in base_seq],
        "rb_up_var": [5e-8 + (v * 1e-9) for v in base_seq],
    }

    return {
        "index_col": "ts",
        "data": data,
        "inference_length": max(1, output_len),
    }


def _update_predict_context(input_len: int, output_len: int, has_df: bool):
    global _predict_input_len, _predict_output_len, _predict_has_df
    try:
        _predict_input_len = max(1, int(input_len))
    except Exception:
        _predict_input_len = 10
    try:
        _predict_output_len = max(1, int(output_len))
    except Exception:
        _predict_output_len = 1
    _predict_has_df = bool(has_df)


def _should_use_cached_predicts() -> bool:
    """INFERENCE CACHING DISABLED: Always generate synthetic payloads with real data.
    
    This ensures every /predict request contains valid model input data,
    regardless of PREDICT_PAYLOAD_MODE or cache flags.
    """
    # Force synthetic mode - never use empty cached payloads
    return False


def _next_predict_payload() -> dict:
    """Generate next prediction payload with unique timestamps.
    
    ALWAYS generates synthetic payloads with real data - caching disabled.
    Every request will contain valid model input with all required features.
    """
    global _predict_payload_seq
    with _predict_payload_lock:
        seq = _predict_payload_seq
        _predict_payload_seq += 1
    # Space out timestamps by output window length to mimic rolling horizon
    # CRITICAL: Use sequential minutes offset to ensure globally unique base times
    base_time = dt.datetime.now(dt.timezone.utc).replace(microsecond=0) + dt.timedelta(minutes=seq * max(1, _predict_output_len))
    
    # Generate synthetic payload with guaranteed unique timestamps
    payload = _build_synthetic_predict_payload(_predict_input_len, _predict_output_len, base_time=base_time)
    
    # Confirm we're generating real data
    DEBUG_ENABLED = os.getenv("DEBUG_LOCUST_PAYLOAD", "0") in {"1", "true", "TRUE"}
    if DEBUG_ENABLED and seq < 3:
        data_keys = list(payload.get("data", {}).keys()) if "data" in payload else []
        data_rows = len(payload.get("data", {}).get("ts", [])) if "data" in payload else 0
        print(f"[LOCUST_PAYLOAD_GEN] seq={seq} mode=SYNTHETIC_ONLY data_keys={data_keys} rows={data_rows}")
    
    # DEBUG: Print payload structure for first few requests
    if seq < 2 and os.getenv("DEBUG_LOCUST_PAYLOAD", "0") == "1":
        import json
        print(f"[LOCUST_DEBUG] Payload seq={seq}:")
        print(f"  index_col: {payload.get('index_col')}")
        print(f"  data keys: {list(payload.get('data', {}).keys())}")
        print(f"  first 3 timestamps: {payload.get('data', {}).get('ts', [])[:3]}")
    
    # Optional: Log first payload for debugging
    if seq == 0 and os.getenv("DEBUG_LOCUST_PAYLOAD", "0") == "1":
        import json
        print(f"[DEBUG] First synthetic payload: {json.dumps(payload, indent=2)[:500]}")
    
    return payload


def _run_preflight_predict_check(environment, input_len: int, output_len: int, has_df: bool):
    if os.getenv("LOCUST_PREFLIGHT_DISABLE", "0") in {"1", "true", "TRUE"}:
        return True
    _update_predict_context(input_len, output_len, has_df)
    payload = _build_synthetic_predict_payload(input_len, output_len)
    status_code = None
    error_text = None
    try:
        import requests
        resp = requests.post(PREDICT_URL, json=payload, timeout=30)
        status_code = getattr(resp, "status_code", None)
        if status_code != 200:
            error_text = None if resp is None else resp.text[:256]
    except Exception as exc:
        error_text = str(exc)
    ok = status_code == 200 and error_text is None
    _append_jsonl({
        "ts": time.time(),
        "event": "predict_prerun_check",
        "status_code": status_code,
        "ok": ok,
        "error": error_text,
    })
    if not ok:
        try:
            environment.process_exit_code = 1
        except Exception:
            pass
        try:
            runner = getattr(environment, "runner", None)
            if runner is not None:
                runner.quit()
        except Exception:
            pass
    return ok

def _is_headless():
    # Locust sets environment.parsed_options.headless when running headless
    try:
        from locust.env import Environment  # type: ignore
        return bool(events.request._handlers and any(True for _ in [1]))  # crude presence check
    except Exception:
        return False

@events.test_start.add_listener
def on_test_start(environment, **kw):  # noqa: D401
    """
    SIMPLIFIED TEST START - No warmup, no health checks, no ping.
    
    Both UI and headless modes start immediately with valid predict requests.
    """
    global _headless_auto_started
    if _headless_auto_started:
        return
    _headless_auto_started = True
    
    _append_jsonl({
        "ts": time.time(),
        "event": "test_start_unified",
        "mode": "UI_AND_HEADLESS_IDENTICAL",
    })
    
    # Mark predict as ready globally
    globals()["_predict_ready"] = True
    
    # Resolve sequence lengths from environment or use defaults
    in_len = int(os.getenv("PREDICT_INPUT_LEN", "10"))
    out_len = int(os.getenv("PREDICT_OUTPUT_LEN", "1"))
    _update_predict_context(in_len, out_len, False)
    
    _append_jsonl({
        "ts": time.time(),
        "event": "predict_config",
        "input_seq_len": in_len,
        "output_seq_len": out_len,
        "payload_mode": "SYNTHETIC_ONLY",
        "caching_disabled": True,
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
    """
    SIMPLIFIED REQUEST LOGGING - All predict requests are logged consistently.
    
    No warmup state checks, no mode-specific filtering.
    """
    try:
        status_code = getattr(response, "status_code", None) if response else None
        
        if name == "predict":
            # Always log predict requests (success or failure)
            log_this = False
            if LOG_PREDICT_ALL:
                log_this = True
            elif status_code and status_code == 200:
                log_this = True
            elif LOG_PREDICT_ERRORS and (exception or (status_code and status_code >= 400)):
                log_this = True
            
            if not log_this:
                return
            
            body_snip = None
            if LOG_PREDICT_RESPONSE_CHARS > 0 and response is not None:
                try:
                    txt = response.text
                    if len(txt) > LOG_PREDICT_RESPONSE_CHARS:
                        body_snip = txt[:LOG_PREDICT_RESPONSE_CHARS] + "..."  # truncated
                    else:
                        body_snip = txt
                except Exception:
                    body_snip = None
            
            payload = None
            if LOG_PREDICT_PAYLOAD and context and isinstance(context, dict):
                payload = context.get("request_json")
            
            rec = {
                "ts": time.time(),
                "request_type": request_type,
                "name": name,
                "response_time_ms": response_time,
                "status_code": status_code,
                "error": str(exception) if exception else None,
            }
            if body_snip is not None:
                rec["response_snip"] = body_snip
            if payload is not None:
                rec["payload"] = payload
            _append_jsonl(rec)
        else:
            # Log all other requests
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
    """
    UNIFIED USER MODEL - Works identically in UI and headless modes.
    
    The host attribute MUST be just the base URL (scheme + netloc) without any path.
    All paths are specified in the request methods (e.g., self.client.post('/predict', ...))
    """
    # Extract just the base URL (no path) for proper Locust HttpUser behavior
    host = _predict_host  # This is already just scheme://netloc from URL parsing above
    wait_time = between(_user_wait_min, _user_wait_max)


    def _download_warmup(self):
        global _download_ready, _download_warm_attempts
        if _download_ready:
            return
        global _download_active_url
        # Use plain requests for warmup so attempts are NOT counted in Locust metrics
        # This avoids polluting the scoreboard with warmup failures/timeouts.
        try:
            import requests  # local import to keep global deps minimal
        except Exception:
            requests = None  # pragma: no cover
        for attempt in range(1, DOWNLOAD_WARMUP_ATTEMPTS + 1):
            _download_warm_attempts = attempt
            ok = False
            chosen = None
            try:
                if requests is None:
                    raise RuntimeError("requests unavailable for warmup")
                r = requests.get(EP_DOWNLOAD, timeout=10)
                if r.status_code == 200:
                    ok = True
                    chosen = EP_DOWNLOAD
                elif EP_DOWNLOAD_ALT:
                    r2 = requests.get(EP_DOWNLOAD_ALT, timeout=10)
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

    def _predict_warmup(self):
        """Perform a single /predict call with a minimal valid payload to seed server cache.

        Strategy:
        - Query /predict_ping to get input/output sequence lengths (fallback to 10/1).
        - Build a tiny DataFrame shape with columns ['time','value'] and enough rows.
        - POST to /predict; on 200, set _predict_ready = True.
        """
        global _predict_ready
        global _warmup_done
        if _predict_ready or _warmup_done:
            return
        # Single-flight warm-up across all users
        if not _warmup_lock.acquire(blocking=False):
            return
        try:
            # Discover predict base for ping
            r = self.client.get(_predict_ping_path, name="predict_ping", timeout=10)
            in_len = 10
            out_len = 1
            has_df = False
            if r is not None and r.status_code == 200:
                try:
                    js = r.json()
                    if isinstance(js, dict):
                        in_len = int(js.get("input_seq_len") or in_len)
                        out_len = int(js.get("output_seq_len") or out_len)
                        has_df = bool(js.get("has_df"))
                except Exception:
                    pass
            if has_df and _predict_cache_enabled:
                try:
                    pr_cached = self.client.post(_predict_request_path, json={}, name="predict_warmup", timeout=60)
                except Exception as exc:
                    pr_cached = None
                    _append_jsonl({
                        "ts": time.time(),
                        "event": "predict_warmup_cached_error",
                        "error": str(exc),
                    })
                else:
                    ok_cached = pr_cached is not None and pr_cached.status_code == 200
                    _append_jsonl({
                        "ts": time.time(),
                        "event": "predict_warmup_result",
                        "status_code": None if pr_cached is None else pr_cached.status_code,
                        "ok": ok_cached,
                        "mode": "cached",
                        "rows": None,
                        "in_len": in_len,
                        "out_len": out_len,
                    })
                    if ok_cached:
                        _update_predict_context(in_len, out_len, has_df)
                        _predict_ready = True
                        _warmup_done = True
                        return
            payload = _build_synthetic_predict_payload(in_len, out_len)
            pr = self.client.post(_predict_request_path, json=payload, name="predict_warmup", timeout=30)
            ok = pr is not None and pr.status_code == 200
            _append_jsonl({
                "ts": time.time(),
                "event": "predict_warmup_result",
                "status_code": None if pr is None else pr.status_code,
                "ok": ok,
                "mode": "synthetic",
                "rows": len(payload.get("data", {}).get("ts", [])),
                "in_len": in_len,
                "out_len": out_len,
            })
            if ok:
                _update_predict_context(in_len, out_len, has_df)
                _predict_ready = True
                _warmup_done = True
        except Exception as e:
            _append_jsonl({"ts": time.time(), "event": "predict_warmup_error", "error": str(e)})
        finally:
            try:
                _warmup_lock.release()
            except Exception:
                pass

    def on_start(self):
        """
        SIMPLIFIED USER INITIALIZATION - No warmup, no ping, no health checks.
        
        Just log the configuration and start sending predict requests immediately.
        This ensures UI and headless modes behave identically.
        """
        global _session_header_written
        if not _session_header_written:
            # Optionally truncate log file to avoid historical noise
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
            "predict_url": PREDICT_URL,
            "predict_host": _predict_host,
            "predict_path": _predict_request_path,
            "predict_warmup_disabled": True,
            "predict_payload_mode": "SYNTHETIC_ONLY",
            "predict_cache_disabled": True,
            "mode": "UNIFIED_UI_AND_HEADLESS",
        })
        
        # Mark predict as ready immediately (no warmup needed)
        globals()["_predict_ready"] = True
        
        # Resolve sequence lengths from environment or use defaults
        in_len = int(os.getenv("PREDICT_INPUT_LEN", "10"))
        out_len = int(os.getenv("PREDICT_OUTPUT_LEN", "1"))
        _update_predict_context(in_len, out_len, False)
        
        # No warmup, no health checks, no download tests - go straight to predict tasks

    @task(100)
    def predict(self):
        """
        UNIFIED PREDICT TASK - Works identically in UI and headless modes.
        
        Always generates valid synthetic payloads with real data.
        No warmup, no ping, no caching - just pure /predict requests.
        """
        payload = _next_predict_payload()
        
        # ===== ENHANCED DIAGNOSTIC LOGGING =====
        global _predict_payload_seq
        DEBUG_ENABLED = os.getenv("DEBUG_LOCUST_PAYLOAD", "0") in {"1", "true", "TRUE"}
        ALWAYS_LOG_FIRST = os.getenv("LOCUST_ALWAYS_LOG_FIRST", "0") in {"1", "true", "TRUE"}  # Changed default to "0"
        
        if (_predict_payload_seq < 5 and ALWAYS_LOG_FIRST) or DEBUG_ENABLED:
            import json
            # Log payload structure
            has_data = bool(payload.get("data"))
            data_keys = list(payload.get("data", {}).keys()) if has_data else []
            
            print(f"\n{'='*80}")
            print(f"[LOCUST_SENDING] seq={_predict_payload_seq} has_data={has_data}")
            print(f"[LOCUST_SENDING] data_keys={data_keys}")
            print(f"[LOCUST_SENDING] inference_length={payload.get('inference_length')}")
            print(f"[LOCUST_SENDING] index_col={payload.get('index_col')}")
            
            # Check for timestamp fields
            if has_data:
                data = payload["data"]
                ts_field = None
                ts_sample = None
                ts_unique = None
                ts_total = None
                
                for field in ["ts", "time", "timestamp", "date"]:
                    if field in data:
                        ts_field = field
                        ts_values = data[field]
                        ts_total = len(ts_values) if isinstance(ts_values, list) else "?"
                        ts_unique = len(set(ts_values)) if isinstance(ts_values, list) else "?"
                        ts_sample = ts_values[:5] if isinstance(ts_values, list) else str(ts_values)[:100]
                        break
                
                if ts_field:
                    print(f"[LOCUST_SENDING] ✅ FOUND timestamp_field='{ts_field}'")
                    print(f"[LOCUST_SENDING]    total={ts_total} unique={ts_unique}")
                    print(f"[LOCUST_SENDING]    sample={ts_sample}")
                else:
                    print(f"[LOCUST_SENDING] ❌ NO timestamp field found in {data_keys}")
            
            # Log raw JSON preview
            json_str = json.dumps(payload, default=str)
            print(f"[LOCUST_SENDING] json_preview={json_str[:500]}...")
            print(f"[LOCUST_SENDING] json_length={len(json_str)} bytes")
            print(f"[LOCUST_SENDING] target_url={_predict_request_path}")
            print(f"{'='*80}\n")
        # ===== END ENHANCED DIAGNOSTIC LOGGING =====
        
        try:
            r = self.client.post(_predict_request_path, json=payload, name="predict")
            
            # Log response status for first few requests
            if (_predict_payload_seq < 5 and ALWAYS_LOG_FIRST) or DEBUG_ENABLED:
                status = r.status_code if r else "NO_RESPONSE"
                print(f"[LOCUST_RESPONSE] seq={_predict_payload_seq} status={status}")
                if r and r.status_code >= 400:
                    print(f"[LOCUST_RESPONSE] error_body={r.text[:300]}")
            
            if LOG_PREDICT_PAYLOAD:
                if not hasattr(r, "context"):
                    try:
                        r.context = {}
                    except Exception:
                        pass
                try:
                    r.context["request_json"] = payload
                except Exception:
                    pass
        except Exception as exc:
            if DEBUG_ENABLED or (_predict_payload_seq < 5 and ALWAYS_LOG_FIRST):
                print(f"[LOCUST_ERROR] seq={_predict_payload_seq} exception={exc}")
            pass

    # ===== REMOVED TASKS - /healthz and /download_processed disabled =====
    # These tasks have been removed to ensure 100% focus on /predict requests.
    # Both UI and headless modes now only execute predict() task.


__all__ = ["PipelineUser"]
if __name__ == "__main__":
    # Allow running directly for quick local debug
    from locust import run_single_user
    run_single_user(PipelineUser)
