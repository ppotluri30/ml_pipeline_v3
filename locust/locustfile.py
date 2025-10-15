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
import os, json, time, threading, uuid, random, math, datetime as dt

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
# Enhanced predict logging controls
LOG_PREDICT_ALL = os.getenv("LOG_PREDICT_ALL", "0") == "1"  # log every predict (success + failure)
LOG_PREDICT_ERRORS = os.getenv("LOG_PREDICT_ERRORS", "1") == "1"  # log failed predict even if not logging all
LOG_PREDICT_RESPONSE_CHARS = int(os.getenv("LOG_PREDICT_RESPONSE_CHARS", "0"))  # capture first N chars of body
LOG_PREDICT_PAYLOAD = os.getenv("LOG_PREDICT_PAYLOAD", "0") == "1"  # echo JSON payload (small tests only)
PREDICT_WARMUP_DISABLE = os.getenv("PREDICT_WARMUP_DISABLE", "0") in {"1", "true", "TRUE"}
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
    base = PREDICT_URL.rsplit('/', 1)[0]
    ping_url = f"{base}/predict_ping"
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

    The column order matches the processed parquet generated by the pipeline
    (10 network metrics, target value, and six time features).
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
    step = max(1, int(freq_minutes))
    times_dt = [t0 + dt.timedelta(minutes=i * step) for i in range(total)]
    times = [ts.replace(tzinfo=None).isoformat() for ts in times_dt]
    base_seq = [float(i % 50) for i in range(total)]
    data = {
        "time": times,
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
        "value": [float((i * 1.5) % 100) for i in range(total)],
    }

    # Time-derived cyclical features
    min_of_day = [(ts.hour * 60 + ts.minute) for ts in times_dt]
    day_of_week = [ts.weekday() for ts in times_dt]
    day_of_year = [ts.timetuple().tm_yday for ts in times_dt]

    def _sin(series, period):
        return [math.sin(val * (2 * math.pi / period)) for val in series]

    def _cos(series, period):
        return [math.cos(val * (2 * math.pi / period)) for val in series]

    data["min_of_day_sin"] = _sin(min_of_day, 1440.0)
    data["min_of_day_cos"] = _cos(min_of_day, 1440.0)
    data["day_of_week_sin"] = _sin(day_of_week, 7.0)
    data["day_of_week_cos"] = _cos(day_of_week, 7.0)
    data["day_of_year_sin"] = _sin(day_of_year, 365.25)
    data["day_of_year_cos"] = _cos(day_of_year, 365.25)

    return {
        "index_col": "time",
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
    if PREDICT_PAYLOAD_MODE == "cached":
        return True
    if PREDICT_PAYLOAD_MODE == "synthetic":
        return False
    return _predict_has_df


def _next_predict_payload() -> dict:
    if _should_use_cached_predicts():
        return {}
    global _predict_payload_seq
    with _predict_payload_lock:
        seq = _predict_payload_seq
        _predict_payload_seq += 1
    # Space out timestamps by output window length to mimic rolling horizon
    base_time = dt.datetime.now(dt.timezone.utc).replace(microsecond=0) + dt.timedelta(minutes=seq * max(1, _predict_output_len))
    return _build_synthetic_predict_payload(_predict_input_len, _predict_output_len, base_time=base_time)


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
    # In headless mode, ensure at least one user executes warm-up immediately.
    global _headless_auto_started
    if _headless_auto_started:
        return
    _headless_auto_started = True
    # Warm-up: ping health endpoint, then sleep briefly before spawning load to avoid early connection churn.
    _append_jsonl({
        "ts": time.time(),
        "event": "headless_test_start",
    })
    try:
        import requests  # local import to avoid global dependency
        base = PREDICT_URL.rsplit('/', 1)[0]
        health_url = EP_HEALTH if EP_HEALTH.startswith("http") else f"{base}{EP_HEALTH}"
        ok = False
        # Quick initial probes (fast-fail if service totally down)
        for i in range(1, 4):
            try:
                resp = requests.get(health_url, timeout=5)
                ok = resp.status_code == 200
            except Exception:
                ok = False
            _append_jsonl({
                "ts": time.time(),
                "event": "warmup_health_attempt",
                "attempt": i,
                "ok": ok
            })
            if ok:
                break
            time.sleep(1.0)

        # If requested, wait until the inference model reports ready via /healthz.model_ready
        # CONTROL via env: LOCUST_WAIT_FOR_MODEL (default: true), and timeout LOCUST_WAIT_FOR_MODEL_TIMEOUT (seconds)
        wait_for_model = os.getenv("LOCUST_WAIT_FOR_MODEL", "1") not in {"0", "false", "FALSE"}
        wait_timeout = float(os.getenv("LOCUST_WAIT_FOR_MODEL_TIMEOUT", "60"))
        if wait_for_model:
            t_deadline = time.time() + wait_timeout
            model_ready = False
            # If earlier quick probes succeeded, still verify model_ready flag
            while time.time() < t_deadline:
                try:
                    resp = requests.get(health_url, timeout=5)
                    if resp is not None and resp.status_code == 200:
                        try:
                            js = resp.json()
                            model_ready = bool(js.get("model_ready"))
                        except Exception:
                            model_ready = False
                    else:
                        model_ready = False
                except Exception:
                    model_ready = False
                _append_jsonl({
                    "ts": time.time(),
                    "event": "warmup_model_ready_check",
                    "model_ready": model_ready,
                    "time_left": max(0.0, round(t_deadline - time.time(), 2))
                })
                if model_ready:
                    break
                time.sleep(1.0)
            if not model_ready:
                _append_jsonl({
                    "ts": time.time(),
                    "event": "warmup_model_ready_timeout",
                    "wait_timeout": wait_timeout
                })

        # Small settle delay regardless of health result
        time.sleep(2.0)

        in_len, out_len, has_df = _resolve_predict_lengths()
        _append_jsonl({
            "ts": time.time(),
            "event": "predict_ping_snapshot",
            "input_seq_len": in_len,
            "output_seq_len": out_len,
            "has_df": has_df,
        })
        if not _run_preflight_predict_check(environment, in_len, out_len, has_df):
            return
    except Exception as e:
        _append_jsonl({
            "ts": time.time(),
            "event": "warmup_error",
            "error": str(e)
        })
    # Optionally trigger a Kafka burst to exercise consumer backpressure
    try:
        if KAFKA_BURST and KAFKA_BURST_COUNT > 0:
            payload = {
                "count": KAFKA_BURST_COUNT,
            }
            if KAFKA_BURST_TTL_MS:
                try:
                    payload["ttl_ms"] = int(KAFKA_BURST_TTL_MS)
                except Exception:
                    pass
            if KAFKA_BURST_KEY_PREFIX:
                payload["key_prefix"] = KAFKA_BURST_KEY_PREFIX
            import requests  # only used here to avoid adding to common path
            resp = requests.post(f"{PREDICT_URL.rsplit('/',1)[0]}/publish_inference_claims", json=payload, timeout=30)
            _append_jsonl({
                "ts": time.time(),
                "event": "kafka_burst_triggered",
                "status_code": getattr(resp, "status_code", None),
                "response": None if resp is None else resp.text[:200],
            })
            # Wait for background inference to run once so cached fast path is available
            try:
                base = PREDICT_URL.rsplit('/', 1)[0]
                ping = f"{base}/predict_ping"
                t_end = time.time() + 6.0
                seen_busy = False
                while time.time() < t_end:
                    try:
                        pr = requests.get(ping, timeout=2)
                        js = pr.json() if pr is not None else {}
                        has_df = bool(js.get("has_df"))
                        busy = bool(js.get("busy"))
                        if busy:
                            seen_busy = True
                        if has_df and (not busy) and seen_busy:
                            _append_jsonl({"ts": time.time(), "event": "post_burst_ready", "has_df": has_df})
                            break
                    except Exception:
                        pass
                    time.sleep(0.25)
            except Exception:
                pass
    except Exception as e:
        _append_jsonl({
            "ts": time.time(),
            "event": "kafka_burst_error",
            "error": str(e),
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
            # If predict not ready, optionally record as skipped
            if not _predict_ready and LOG_PREDICT_ALL:
                _append_jsonl({
                    "ts": time.time(),
                    "request_type": request_type,
                    "name": name,
                    "skipped": True,
                    "reason": "predict_not_ready"
                })
                return
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
            base = PREDICT_URL.rsplit('/', 1)[0]
            ping_url = f"{base}/predict_ping"
            r = self.client.get(ping_url, name="predict_ping", timeout=10)
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
            if has_df:
                try:
                    pr_cached = self.client.post(PREDICT_URL, json={}, name="predict_warmup", timeout=60)
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
            pr = self.client.post(PREDICT_URL, json=payload, name="predict_warmup", timeout=30)
            ok = pr is not None and pr.status_code == 200
            _append_jsonl({
                "ts": time.time(),
                "event": "predict_warmup_result",
                "status_code": None if pr is None else pr.status_code,
                "ok": ok,
                "mode": "synthetic",
                "rows": len(payload.get("data", {}).get("time", [])),
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
            "predict_payload_mode": PREDICT_PAYLOAD_MODE,
        })
        # Warm health once (ignored errors)
        try:
            self.client.get(EP_HEALTH, name="healthz-warm")
        except Exception:
            pass
        # Perform predict warm-up once globally to seed cache and avoid 400s for empty-body requests
        if PREDICT_WARMUP_DISABLE:
            # Skip warmup entirely (assumes service.df is preloaded via claim-check)
            # If already warmed, mark ready for this user
            in_len, out_len, has_df = _resolve_predict_lengths()
            _update_predict_context(in_len, out_len, has_df)
            globals()["_predict_ready"] = True
        elif not _warmup_done:
            self._predict_warmup()
        else:
            globals()["_predict_ready"] = True
        self._download_warmup()
        # Per-user ramp delay to avoid initial 429s and transport churn
        try:
            delay = random.uniform(5.0, 10.0)
            _append_jsonl({
                "ts": time.time(),
                "event": "user_ramp_delay",
                "seconds": round(delay, 3)
            })
            time.sleep(delay)
        except Exception:
            pass

    @task(80)
    def predict(self):
        if not _predict_ready:
            # Try a lightweight warm-up retry in case this user started before cache was primed
            self._predict_warmup()
            return
        payload = _next_predict_payload()
        try:
            r = self.client.post(PREDICT_URL, json=payload, name="predict")
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
        except Exception:
            pass

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
