"""FastAPI inference API serving synchronous inference requests."""
from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from data_utils import strip_timezones, time_to_feature
from trace_utils import trace_df_operation, trace_dataframe, trace_operation, trace_error, TRACE_ENABLED

InferenceHTTPError = None  # Backwards compatibility shim; local pool deprecated

# Prometheus metrics (optional dependency)
try:
    from prometheus_client import Gauge, Counter, Histogram, start_http_server, generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    _PROMETHEUS_AVAILABLE = True
except Exception:
    _PROMETHEUS_AVAILABLE = False

if _PROMETHEUS_AVAILABLE:
    QUEUE_LEN = Gauge("inference_queue_len", "Current queue size")
    ACTIVE_WORKERS = Gauge("inference_active_workers", "Running worker tasks")
    WORKERS_TOTAL = Gauge("inference_workers_total", "Configured worker slots")
    WORKERS_BUSY = Gauge("inference_workers_busy", "Workers currently processing jobs")
    WORKERS_IDLE = Gauge("inference_workers_idle", "Workers currently idle")
    WORKER_UTILIZATION = Gauge("inference_worker_utilization", "Busy worker ratio (0-1)")
    QUEUE_WAIT_LATEST = Gauge("inference_queue_wait_latest_seconds", "Queue wait time of the most recent job in seconds")
    INFERENCE_DURATION_LATEST = Gauge("inference_latency_latest_seconds", "Duration of the most recent inference execution in seconds")
    QUEUE_OLDEST_WAIT = Gauge("inference_queue_oldest_wait_seconds", "Oldest queued job age in seconds")
    JOBS_PROCESSED = Counter("inference_jobs_processed_total", "Total processed jobs")
    JOB_OUTCOME = Counter("inference_jobs_outcome_total", "Total jobs by terminal outcome", ["outcome"])
    QUEUE_WAIT_TIME = Histogram(
        "inference_queue_wait_seconds",
        "Seconds jobs spent waiting in queue",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
    )
    INFERENCE_LATENCY = Histogram(
        "inference_latency_seconds",
        "Seconds spent executing inference for a job",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
    )
    MODEL_READY = Gauge("inference_model_ready", "Whether model is ready (1=yes,0=no)")
else:
    # Fallback no-op objects to avoid guarding metrics everywhere
    class _NoopMetric:
        def labels(self, *a, **k):
            return self

        def set(self, *a, **k):
            return

        def inc(self, *a, **k):
            return

        def observe(self, *a, **k):
            return

    QUEUE_LEN = _NoopMetric()
    ACTIVE_WORKERS = _NoopMetric()
    WORKERS_TOTAL = _NoopMetric()
    WORKERS_BUSY = _NoopMetric()
    WORKERS_IDLE = _NoopMetric()
    WORKER_UTILIZATION = _NoopMetric()
    QUEUE_WAIT_LATEST = _NoopMetric()
    INFERENCE_DURATION_LATEST = _NoopMetric()
    QUEUE_OLDEST_WAIT = _NoopMetric()
    MODEL_READY = _NoopMetric()
    JOBS_PROCESSED = _NoopMetric()
    JOB_OUTCOME = _NoopMetric()
    QUEUE_WAIT_TIME = _NoopMetric()
    INFERENCE_LATENCY = _NoopMetric()

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


# Legacy compatibility knobs (no longer used for concurrency now that inference is synchronous)
_DEFAULT_CONCURRENCY = max(1, os.cpu_count() or 1)
PREDICT_MAX_CONCURRENCY = max(1, _env_int("PREDICT_MAX_CONCURRENCY", _DEFAULT_CONCURRENCY))
_DEFAULT_THREADPOOL = max(PREDICT_MAX_CONCURRENCY, _DEFAULT_CONCURRENCY)
PREDICT_THREADPOOL_WORKERS = max(1, _env_int("PREDICT_THREADPOOL_WORKERS", _DEFAULT_THREADPOOL))
QUEUE_WORKERS = PREDICT_MAX_CONCURRENCY
QUEUE_MAXSIZE = 1

# Startup / readiness behaviour
WAIT_FOR_MODEL = _env_flag("WAIT_FOR_MODEL", True)
MODEL_WAIT_TIMEOUT = float(os.getenv("MODEL_WAIT_TIMEOUT", "120"))
PREWARM_ENABLED = _env_flag("ENABLE_PREWARM", False)

# Misc feature toggles
ENABLE_PREDICT_CACHE_DEFAULT = _env_flag("ENABLE_PREDICT_CACHE", True)


def _start_resource_logger_if_enabled() -> None:
    """Optionally emit periodic CPU/memory usage logs when enabled via env."""

    try:
        if not _env_flag("ENABLE_RESOURCE_LOGS", False):
            return
        try:
            import psutil  # type: ignore
        except Exception:
            print({"service": "inference", "event": "resource_logger_psutil_missing"})
            return

        def _log_resource_usage() -> None:
            try:
                proc = psutil.Process()
                proc.cpu_percent(interval=0.0)  # prime measurement
                while True:
                    cpu = proc.cpu_percent(interval=0.1)
                    rss_mb = proc.memory_info().rss / (1024 * 1024)
                    print(
                        {
                            "service": "inference",
                            "event": "resource_usage",
                            "cpu_percent": cpu,
                            "mem_mb": round(rss_mb, 1),
                        }
                    )
                    time.sleep(10)
            except Exception as exc:  # pragma: no cover - best-effort logging
                print({"service": "inference", "event": "resource_logger_error", "error": str(exc)})

        threading.Thread(target=_log_resource_usage, daemon=True, name="resource-logger").start()
    except Exception:
        pass


app = FastAPI(title="Inference Synchronous API")


@app.post("/scale_workers")
async def scale_workers(payload: dict = Body(...)):
    """Legacy endpoint retained for compatibility. Pool-based scaling is deprecated."""
    _queue_log(
        "scale_workers_deprecated",
        requested_workers=payload.get("workers"),
        note="Local process pool removed; adjust concurrency via external clients",
    )
    return {
        "status": "deprecated",
        "message": "Local pool queue deprecated — concurrency now managed by distributed Locust workers.",
        "workers": PREDICT_MAX_CONCURRENCY,
    }
ENABLE_PUBLISH_API = os.getenv("ENABLE_PUBLISH_API", "0").lower() in {"1", "true", "yes"}
_publish_producer = None  # lazy-init if endpoint used
_publish_topic = os.getenv("PUBLISH_TOPIC", os.getenv("CONSUMER_TOPIC_0", "inference-data"))

_startup_epoch = time.time()
_startup_ready_ms: float | None = None


_pool_started = False
_queue_monitor_task: asyncio.Task | None = None
_queue_monitor_stop_event: asyncio.Event | None = None
_event_loop_monitor_task: asyncio.Task | None = None

_CONCURRENCY_SEMAPHORE: asyncio.Semaphore | None = None
_concurrency_semaphore_lock = threading.Lock()

_threadpool_executor: ThreadPoolExecutor | None = None
_threadpool_executor_lock = threading.Lock()

queue_metrics = {
    "enqueued": 0,
    "active": 0,
    "completed": 0,
    "served_cached": 0,
    "last_duration_ms": 0,
    "last_wait_ms": 0,
    "max_wait_ms": 0,
    "max_exec_ms": 0,
    "total_wait_ms": 0.0,
    "wait_samples": 0,
    "total_exec_ms": 0.0,
    "exec_samples": 0,
    "last_prep_ms": 0,
    "max_prep_ms": 0,
    "total_prep_ms": 0.0,
    "prep_samples": 0,
    # error metrics
    "error_500_total": 0,
    "last_error_type": None,
}

queue_metrics_lock = threading.Lock()

event_loop_stats = {
    "samples": 0,
    "total_ms": 0.0,
    "max_ms": 0.0,
    "last_ms": 0.0,
}

event_loop_stats_lock = threading.Lock()


def _get_concurrency_semaphore() -> asyncio.Semaphore:
    global _CONCURRENCY_SEMAPHORE
    if _CONCURRENCY_SEMAPHORE is None:
        with _concurrency_semaphore_lock:
            if _CONCURRENCY_SEMAPHORE is None:
                _CONCURRENCY_SEMAPHORE = asyncio.Semaphore(PREDICT_MAX_CONCURRENCY)
    return _CONCURRENCY_SEMAPHORE

def _get_threadpool_executor() -> ThreadPoolExecutor:
    global _threadpool_executor
    if _threadpool_executor is None:
        with _threadpool_executor_lock:
            if _threadpool_executor is None:
                _threadpool_executor = ThreadPoolExecutor(
                    max_workers=PREDICT_THREADPOOL_WORKERS,
                    thread_name_prefix="predict-exec",
                )
    return _threadpool_executor

# (Removed rolling inference duration tracking in rollback)


def _cache_enabled() -> bool:
    raw = os.getenv("ENABLE_PREDICT_CACHE")
    if raw is None:
        return ENABLE_PREDICT_CACHE_DEFAULT
    return raw.lower() in {"1", "true", "yes"}

# --------------- Fallback MLflow latest-model loader -----------------
def _fallback_load_latest_model(reason: str = "startup") -> bool:
    """Attempt to load the most recent usable MLflow run when no promotion manifest exists.

    Selection logic:
      1. Try runs (across all experiments) with tag promoted=true ordered by end_time DESC.
      2. Fallback: any FINISHED runs ordered by end_time DESC.
    A run is considered usable if it contains a model artifact folder named after its param 'model_type'.

    On success: sets inferencer.current_model and related metadata, returns True.
    On failure: returns False (logs structured events for observability).
    """
    try:  # noqa: C901 (keep logic linear & explicit)
        from main import inferencer as _inf, _enrich_loaded_model  # type: ignore
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow import pyfunc
        client = MlflowClient()
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        # Enumerate experiments
        try:
            if hasattr(client, "search_experiments"):
                experiments = client.search_experiments()
            elif hasattr(client, "list_experiments"):
                experiments = client.list_experiments()
            else:
                experiments = []
        except Exception as ee:  # noqa: BLE001
            experiments = []
            print({"service": "inference", "event": "fallback_experiments_enum_fail", "error": str(ee)})
        exp_ids = [e.experiment_id for e in experiments]
        # Provide a minimal fallback if enumeration failed
        if not exp_ids:
            exp_ids = ["0"]  # 'Default' typical id
        print({"service": "inference", "event": "fallback_search_start", "reason": reason, "experiments": exp_ids})

        def _search(filter_string: str):
            try:
                return client.search_runs(
                    experiment_ids=exp_ids,
                    filter_string=filter_string,
                    order_by=["attributes.end_time DESC"],
                    max_results=50,
                )
            except Exception as se:  # noqa: BLE001
                print({"service": "inference", "event": "fallback_search_fail", "filter": filter_string, "error": str(se)})
                return []

        # Phase 1: explicitly prefer promoted=true tagged runs
        promoted_runs = _search("tags.promoted = 'true' and attributes.status = 'FINISHED'")
        chosen = None
        if promoted_runs:
            chosen = promoted_runs[0]
            try:
                pr_params = chosen.data.params or {}
                pr_cfg_hash = pr_params.get("config_hash") or pr_params.get("CONFIG_HASH")
                print({"service": "inference", "event": "fallback_model_load_promoted", "run_id": chosen.info.run_id, "model_type": pr_params.get("model_type"), "config_hash": pr_cfg_hash})
            except Exception:
                pass
        # Phase 2: fallback to any finished run if no promoted present
        if chosen is None:
            any_runs = _search("attributes.status = 'FINISHED'")
            if not any_runs:
                print({"service": "inference", "event": "fallback_no_runs_found"})
                return False
            chosen = any_runs[0]

        # 'chosen' now references the run to attempt loading
        run_id = chosen.info.run_id
        params = chosen.data.params or {}
        model_type = params.get("model_type") or params.get("MODEL_TYPE") or "model"
        config_hash = params.get("config_hash") or params.get("CONFIG_HASH")
        model_uri_candidates = [f"runs:/{run_id}/{model_type}"]
        if model_type != "model":  # add generic fallback path
            model_uri_candidates.append(f"runs:/{run_id}/model")
        loaded = False
        for cand in model_uri_candidates:
            try:
                print({"service": "inference", "event": "fallback_model_load_attempt", "model_uri": cand, "run_id": run_id})
                mdl = pyfunc.load_model(cand)
                _inf.current_model = mdl
                _inf.current_run_id = run_id
                _inf.current_run_name = model_type
                _inf.model_type = model_type
                _inf.current_config_hash = config_hash
                # enrich (sequence lengths, class, etc.)
                try:
                    _enrich_loaded_model(_inf, run_id, model_type)
                except Exception as enrich_err:  # noqa: BLE001
                    print({"service": "inference", "event": "fallback_enrich_fail", "error": str(enrich_err)})
                print({"service": "inference", "event": "startup_model_fallback_loaded", "run_id": run_id, "model_type": model_type, "config_hash": config_hash})
                loaded = True
                break
            except Exception as load_err:  # noqa: BLE001
                print({"service": "inference", "event": "fallback_model_load_fail", "candidate": cand, "error": str(load_err)})
        return loaded
    except Exception as e:  # noqa: BLE001
        print({"service": "inference", "event": "fallback_unhandled_error", "error": str(e)})
        return False

class PredictRequest(BaseModel):
    inference_length: Optional[int] = Field(None, ge=1, le=10000)
    data: Optional[Dict[str, List[Any]]] = None
    index_col: Optional[str] = None
    if hasattr(BaseModel, "model_config"):
        model_config = {"extra": "allow"}

def _queue_log(event: str, **extra):  # central helper for structured logs
    try:
        payload = {"service": "inference", "event": event, "source": "api"}
        payload.update(extra)
        print(payload, flush=True)
    except Exception:
        pass


def _safe_queue_size() -> int:
    return 0


async def _start_queue_monitor():
    _queue_log("queue_monitor_disabled")


async def _stop_queue_monitor():
    return


TIME_FEATURE_COLUMNS = {
    "min_of_day_sin",
    "min_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
}


async def _monitor_event_loop_lag(interval: float = 0.5) -> None:
    """Track event-loop lag via periodic probes."""
    loop = asyncio.get_running_loop()
    expected = loop.time() + interval
    try:
        while True:
            await asyncio.sleep(interval)
            now = loop.time()
            lag = max(0.0, now - expected)
            expected = now + interval
            lag_ms = lag * 1000.0
            with event_loop_stats_lock:
                event_loop_stats["samples"] += 1
                event_loop_stats["total_ms"] += lag_ms
                event_loop_stats["last_ms"] = lag_ms
                if lag_ms > event_loop_stats["max_ms"]:
                    event_loop_stats["max_ms"] = lag_ms
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        _queue_log("event_loop_monitor_error", error=str(exc))


def _ensure_event_loop_monitor_started() -> None:
    global _event_loop_monitor_task
    if _event_loop_monitor_task is not None and not _event_loop_monitor_task.done():
        return
    try:
        _event_loop_monitor_task = asyncio.create_task(_monitor_event_loop_lag())
    except Exception:
        _queue_log("event_loop_monitor_start_failed")


def _expected_feature_columns(service: Any | None) -> set[str]:
    """Return the set of base feature columns required for inference (excluding time features)."""

    columns: set[str] = set()
    if service is not None:
        try:
            base_df = getattr(service, "df", None)
            if isinstance(base_df, pd.DataFrame) and not base_df.empty:
                columns.update(base_df.columns.tolist())
        except Exception:
            pass
        extra = getattr(service, "expected_feature_columns", None)
        if extra:
            try:
                columns.update(set(extra))
            except Exception:
                pass
    return {col for col in columns if col not in TIME_FEATURE_COLUMNS}


def _format_missing_columns_error(exc: KeyError) -> str:
    raw = exc.args[0] if exc.args else ""
    if isinstance(raw, str):
        start = raw.find("[")
        end = raw.find("]", start + 1)
        if start != -1 and end != -1 and end > start:
            cols_fragment = raw[start + 1 : end]
            columns = [col.strip().strip("'\"") for col in cols_fragment.split(",") if col.strip()]
            if columns:
                return ", ".join(columns)
        cleaned = raw.replace("not found in axis", "").strip(" ' ")
        if cleaned:
            return cleaned
    return str(exc)


@trace_df_operation
def _prepare_dataframe_for_inference(
    req_obj: PredictRequest,
    service: Any | None,
) -> tuple[pd.DataFrame, List[str]]:
    """Normalize inbound request payload into a dataframe suitable for inference.

    Returns the prepared dataframe plus the list of required base feature columns enforced.
    Raises HTTPException(400) for validation failures so the caller can short-circuit.
    """
    
    trace_operation("prepare_start", func="_prepare_dataframe_for_inference")

    data = getattr(req_obj, "data", None)
    if not data:
        raise HTTPException(status_code=400, detail="Request payload must include a non-empty 'data' object")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="'data' must be an object mapping column names to arrays")

    try:
        df_tmp = pd.DataFrame(data)
        trace_dataframe("after_dataframe_creation", df_tmp, {"data_keys": list(data.keys())}, "_prepare_dataframe_for_inference")
    except Exception as exc:  # noqa: BLE001
        trace_error("_prepare_dataframe_for_inference", exc, stage="dataframe_creation")
        raise HTTPException(status_code=400, detail=f"Failed to interpret 'data' payload: {exc}") from exc

    if df_tmp.empty:
        raise HTTPException(status_code=400, detail="Request data must include at least one row")

    index_candidates: List[str] = []
    if req_obj.index_col:
        index_candidates.append(req_obj.index_col)
    for candidate in ("ts", "timestamp", "time", "date"):
        if candidate in df_tmp.columns and candidate not in index_candidates:
            index_candidates.append(candidate)
    if not index_candidates:
        raise HTTPException(status_code=400, detail="Request must include a timestamp column (index_col or ts/time/timestamp)")

    assigned_index = False
    for candidate in index_candidates:
        if candidate not in df_tmp.columns:
            continue
        try:
            # Log raw timestamp values for debugging
            raw_values = df_tmp[candidate].head(5).tolist()
            print(f"[TIMESTAMP_PARSE] Parsing column '{candidate}': sample={raw_values}")
            trace_operation("before_to_datetime", func="_prepare_dataframe_for_inference", 
                          candidate=candidate, raw_sample=raw_values, total_rows=len(df_tmp))
            
            idx = pd.to_datetime(df_tmp[candidate], errors="coerce")
            
            # Log parsed results
            parsed_sample = idx.head(5).tolist()
            unique_count = idx.nunique()
            print(f"[TIMESTAMP_PARSE] Parsed result: unique={unique_count}/{len(idx)} sample={parsed_sample}")
            trace_operation("after_to_datetime", func="_prepare_dataframe_for_inference",
                          candidate=candidate, unique_count=int(unique_count), total=len(idx),
                          parsed_sample=[str(ts) for ts in parsed_sample])
        except Exception as e:
            print(f"[TIMESTAMP_PARSE] Failed to parse '{candidate}': {e}")
            trace_error("_prepare_dataframe_for_inference", e, stage="to_datetime", candidate=candidate)
            continue
        if idx.isna().any():
            raise HTTPException(status_code=400, detail=f"Column '{candidate}' contains invalid timestamps")
        df_tmp = df_tmp.drop(columns=[candidate])
        df_tmp.index = idx
        trace_dataframe("after_set_index", df_tmp, {"candidate": candidate}, "_prepare_dataframe_for_inference")
        assigned_index = True
        break

    if not assigned_index or not isinstance(df_tmp.index, pd.DatetimeIndex):
        raise HTTPException(status_code=400, detail="Unable to determine a valid datetime index from request data")

    df_tmp = df_tmp.sort_index()
    trace_dataframe("after_sort_index", df_tmp, {}, "_prepare_dataframe_for_inference")
    
    df_tmp, _ = strip_timezones(df_tmp)
    trace_dataframe("after_strip_timezones", df_tmp, {}, "_prepare_dataframe_for_inference")
    
    if not isinstance(df_tmp.index, pd.DatetimeIndex):
        raise HTTPException(status_code=400, detail="Index must be datetime after timezone normalization")
    if df_tmp.empty:
        raise HTTPException(status_code=400, detail="Request data must include at least one row after normalization")
    
    # CRITICAL: Detect duplicate timestamps that would cause "Time frequency is zero" errors
    unique_timestamps = df_tmp.index.nunique()
    total_rows = len(df_tmp)
    trace_operation("duplicate_check", func="_prepare_dataframe_for_inference",
                  unique_timestamps=int(unique_timestamps), total_rows=int(total_rows))
    
    if unique_timestamps == 1 and total_rows > 1:
        trace_error("_prepare_dataframe_for_inference", 
                   ValueError(f"All timestamps identical: {df_tmp.index[0]}"),
                   stage="duplicate_validation", unique_timestamps=int(unique_timestamps), total_rows=int(total_rows))
        raise HTTPException(
            status_code=400, 
            detail=f"All {total_rows} timestamps are identical ({df_tmp.index[0]}). Expected unique timestamps for time-series inference."
        )

    conversion_failures: List[str] = []
    for column in df_tmp.columns:
        try:
            df_tmp[column] = pd.to_numeric(df_tmp[column], errors="raise")
        except Exception:
            conversion_failures.append(column)
    if conversion_failures:
        raise HTTPException(status_code=400, detail=f"Columns contain non-numeric values: {', '.join(sorted(conversion_failures))}")

    required_base = _expected_feature_columns(service)
    if required_base:
        missing_base = [col for col in required_base if col not in df_tmp.columns]
        if missing_base:
            raise HTTPException(status_code=400, detail=f"Missing required feature columns: {', '.join(sorted(missing_base))}")

    df_prepared = time_to_feature(df_tmp)
    trace_dataframe("after_time_to_feature", df_prepared, {}, "_prepare_dataframe_for_inference")
    
    return df_prepared, sorted(required_base)


def _current_model_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    try:
        inf = _get_inferencer()
        snapshot = {
            "run_id": getattr(inf, "current_run_id", None),
            "model_type": getattr(inf, "model_type", None),
            "config_hash": getattr(inf, "current_config_hash", None),
            "model_uri": getattr(inf, "current_model_uri", None),
        }
    except Exception:
        snapshot = {}
    snapshot["timestamp"] = time.time()
    return snapshot


def _refresh_prometheus_metrics(duration_s: Optional[float] = None) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    try:
        with queue_metrics_lock:
            active_now = max(0, queue_metrics.get("active", 0))
        total_workers = max(1, PREDICT_MAX_CONCURRENCY)
        busy_workers = min(active_now, total_workers)
        idle_workers = max(0, total_workers - busy_workers)
        WORKERS_TOTAL.set(total_workers)
        ACTIVE_WORKERS.set(busy_workers)
        WORKERS_BUSY.set(busy_workers)
        WORKERS_IDLE.set(idle_workers)
        WORKER_UTILIZATION.set((busy_workers / total_workers) if total_workers else 0.0)
        QUEUE_LEN.set(0)
        QUEUE_WAIT_LATEST.set(0.0)
        QUEUE_OLDEST_WAIT.set(0.0)
        if duration_s is not None:
            INFERENCE_LATENCY.observe(max(0.0, duration_s))
            INFERENCE_DURATION_LATEST.set(max(0.0, duration_s))
        try:
            inf = _get_inferencer()
            MODEL_READY.set(1 if getattr(inf, "current_model", None) is not None else 0)
        except Exception:
            MODEL_READY.set(0)
    except Exception:
        pass


def _ensure_process_pool_ready(snapshot: Dict[str, Any]) -> None:
    _queue_log("process_pool_disabled", snapshot_requested=bool(snapshot))
    _refresh_prometheus_metrics()

# --- Middleware to log ALL requests early (captures 422 JSON errors) ---
@app.middleware("http")
async def log_raw_request(request: Request, call_next):  # type: ignore
    try:
        print({
            "service": "inference",
            "event": "http_request_in",
            "method": request.method,
            "path": request.url.path,
        })
    except Exception:
        pass
    response = await call_next(request)
    return response

@app.get("/healthz")
def healthz():
    # Return basic liveness plus model readiness so callers can decide if warm-up is safe
    try:
        inf = _get_inferencer()
        model_ready = inf.current_model is not None
    except Exception:
        model_ready = False
    qsize = _safe_queue_size()
    return {"status": "ok", "service": "inference-api", "model_ready": model_ready, "queue_length": qsize, "startup_ready_ms": _startup_ready_ms}


@app.get("/ready")
def ready():
    """Readiness endpoint usable by orchestrators/healthchecks.

    Returns 200 only when a model is loaded (model_ready==True). Returns 503 otherwise.
    """
    try:
        inf = _get_inferencer()
        model_ready = inf.current_model is not None
    except Exception:
        model_ready = False
    if model_ready:
        return Response(content=json.dumps({"status": "ready"}), media_type="application/json", status_code=200)
    return Response(content=json.dumps({"status": "not_ready"}), media_type="application/json", status_code=503)

def _get_inferencer():
    # Import the shared Inferencer instance from main without triggering
    # main's runtime start side-effects. main._start_runtime is scheduled
    # from the FastAPI startup handler below to ensure the webserver binds
    # immediately and model-loading happens in background.
    from main import inferencer  # type: ignore
    return inferencer

async def _prewarm_if_needed():  # pragma: no cover (performance side-effect)
    if not PREWARM_ENABLED:
        return
    try:
        inf = _get_inferencer()
        if getattr(inf, "current_model", None) is None:
            return
        # Skip if we already prewarmed this run id
        if hasattr(inf, "_last_prewarm_run_id") and getattr(inf, "_last_prewarm_run_id") == getattr(inf, "current_run_id", None):
            return
        t0 = time.time()
        # PyTorch models: run a minimal inference using existing df if available
        if getattr(inf, "model_class", "").lower() == "pytorch" and inf.df is not None:
            try:
                # Use perform_inference in a thread to avoid blocking loop
                await asyncio.to_thread(inf.perform_inference, inf.get_df_copy(), 1)
            except Exception as ie:  # noqa: BLE001
                _queue_log("prewarm_fail", error=str(ie))
        # Prophet / statsforecast usually compile lazily; trigger generic perform_inference
        elif getattr(inf, "model_class", "").lower() in {"prophet", "statsforecast"} and inf.df is not None:
            try:
                await asyncio.to_thread(inf.perform_inference, inf.get_df_copy(), 1)
            except Exception as ie:  # noqa: BLE001
                _queue_log("prewarm_fail", error=str(ie))
        setattr(inf, "last_prewarm_ms", int((time.time() - t0) * 1000))
        setattr(inf, "_last_prewarm_run_id", getattr(inf, "current_run_id", None))
        _queue_log("prewarm_complete", run_id=getattr(inf, "current_run_id", None), ms=getattr(inf, "last_prewarm_ms", None))
    except Exception as e:  # noqa: BLE001
        _queue_log("prewarm_wrapper_fail", error=str(e))

@app.get("/predict_ping")
def predict_ping():
    try:
        inf = _get_inferencer()
        active_jobs = getattr(inf, "active_inference_jobs", 0)
        return {
            "status": "ok",
            "model_loaded": inf.current_model is not None,
            "has_df": inf.df is not None,
            "busy": active_jobs > 0,
            "active_jobs": active_jobs,
            "input_seq_len": getattr(inf, "input_seq_len", None),
            "output_seq_len": getattr(inf, "output_seq_len", None),
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}

@app.get("/metrics")
def metrics():
    """Lightweight live metrics snapshot (<5ms target)."""
    inf = None
    try:
        inf = _get_inferencer()
    except Exception:
        pass
    build_version = os.getenv("INFER_VERSION")
    with queue_metrics_lock:
        qm_snapshot = queue_metrics.copy()
    with event_loop_stats_lock:
        loop_snapshot = event_loop_stats.copy()
    avg_wait_ms = qm_snapshot["total_wait_ms"] / qm_snapshot["wait_samples"] if qm_snapshot["wait_samples"] else 0.0
    avg_exec_ms = qm_snapshot["total_exec_ms"] / qm_snapshot["exec_samples"] if qm_snapshot["exec_samples"] else 0.0
    avg_prep_ms = qm_snapshot["total_prep_ms"] / qm_snapshot["prep_samples"] if qm_snapshot["prep_samples"] else 0.0
    loop_avg_ms = loop_snapshot["total_ms"] / loop_snapshot["samples"] if loop_snapshot["samples"] else 0.0
    return {
        "mode": "synchronous",
        "queue_length": 0,
        "workers": PREDICT_MAX_CONCURRENCY,
        "active": qm_snapshot["active"],
        "active_jobs": getattr(inf, "active_inference_jobs", None) if inf else None,
        "completed": qm_snapshot["completed"],
        "error_500_total": qm_snapshot["error_500_total"],
        "last_error_type": qm_snapshot["last_error_type"],
        "model_loaded": bool(getattr(inf, "current_model", None)) if inf else False,
        "current_model_hash": getattr(inf, "current_config_hash", None) if inf else None,
        "current_run_id": getattr(inf, "current_run_id", None) if inf else None,
        "current_model_type": getattr(inf, "model_type", None) if inf else None,
        "startup_latency_ms": _startup_ready_ms,
        "prewarm_latency_ms": getattr(inf, "last_prewarm_ms", None) if inf else None,
        "last_inference_duration_ms": qm_snapshot["last_duration_ms"],
        "last_wait_ms": qm_snapshot["last_wait_ms"],
        "max_wait_ms": qm_snapshot["max_wait_ms"],
        "avg_wait_ms": round(avg_wait_ms, 2),
        "max_exec_ms": qm_snapshot["max_exec_ms"],
        "avg_exec_ms": round(avg_exec_ms, 2),
        "last_prep_ms": qm_snapshot["last_prep_ms"],
        "max_prep_ms": qm_snapshot["max_prep_ms"],
        "avg_prep_ms": round(avg_prep_ms, 2),
        "served_cached": qm_snapshot["served_cached"],
        "event_loop_lag_last_ms": round(loop_snapshot["last_ms"], 3),
        "event_loop_lag_max_ms": round(loop_snapshot["max_ms"], 3),
        "event_loop_lag_avg_ms": round(loop_avg_ms, 3),
        "build_version": build_version,
        "status": "ok",
    }

@app.get("/prometheus")
def prometheus_metrics():
    """Prometheus-compatible metrics endpoint (text/plain format)."""
    if not _PROMETHEUS_AVAILABLE:
        return Response(content="# Prometheus client not available\n", media_type="text/plain")
    
    metrics_output = generate_latest()
    return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)

@app.post("/reload_latest")
async def reload_latest():  # pragma: no cover (operational endpoint)
    """Manually trigger fallback load of newest MLflow run (without promotion manifest).

    Returns JSON describing outcome. Always returns 200 even on failure (status field indicates state).
    """
    loaded = await asyncio.to_thread(_fallback_load_latest_model, "manual")
    inf = None
    try:
        inf = _get_inferencer()
    except Exception:  # noqa: BLE001
        pass
    try:
        if _PROMETHEUS_AVAILABLE:
            MODEL_READY.set(1 if loaded else 0)
    except Exception:
        pass
    return {
        "status": "loaded" if loaded else "not_loaded",
        "run_id": getattr(inf, "current_run_id", None) if inf else None,
        "model_type": getattr(inf, "model_type", None) if inf else None,
        "config_hash": getattr(inf, "current_config_hash", None) if inf else None,
    }

@app.post("/predict")
async def predict(
    req: PredictRequest | None = Body(default={}),
    inference_length: int | None = Query(default=None, ge=1, le=10000),
):
    req_id = uuid.uuid4().hex[:8]
    wait_ms: Optional[float] = None
    
    # DEBUG: Log what we received
    if os.getenv("DEBUG_PAYLOAD_TRACE", "0") == "1":
        print(f"[PREDICT_DEBUG] req_id={req_id}: req type={type(req)}", flush=True)
        print(f"[PREDICT_DEBUG] req_id={req_id}: req is None? {req is None}", flush=True)
        if req is not None:
            print(f"[PREDICT_DEBUG] req_id={req_id}: has data attr? {hasattr(req, 'data')}", flush=True)
            data_val = getattr(req, "data", "ATTR_MISSING")
            print(f"[PREDICT_DEBUG] req_id={req_id}: data value type={type(data_val)}, is None={data_val is None}", flush=True)
            if data_val and data_val != "ATTR_MISSING":
                print(f"[PREDICT_DEBUG] req_id={req_id}: data keys={list(data_val.keys()) if isinstance(data_val, dict) else 'NOT_DICT'}", flush=True)
                if isinstance(data_val, dict) and "timestamp" in data_val:
                    ts_sample = data_val["timestamp"][:3] if isinstance(data_val.get("timestamp"), list) else "NOT_LIST"
                    print(f"[PREDICT_DEBUG] req_id={req_id}: timestamp sample={ts_sample}", flush=True)
    try:
        if _cache_enabled() and ((req is None) or ((not getattr(req, "data", None)) and inference_length is None and getattr(req, "inference_length", None) is None)):
            inf_cached = _get_inferencer()
            # Use thread-safe accessor when available
            if hasattr(inf_cached, "get_last_prediction_copy"):
                last_response = inf_cached.get_last_prediction_copy()
            else:
                last_response = getattr(inf_cached, "last_prediction_response", None)
            if last_response:
                cached = last_response.copy()
                cached["status"] = "SUCCESS_CACHED"
                cached["cached"] = True
                cached["req_id"] = req_id
                with queue_metrics_lock:
                    queue_metrics["served_cached"] += 1
                    served_cached = queue_metrics["served_cached"]
                _queue_log("predict_served_cached_direct", req_id=req_id, served_cached=served_cached)
                return cached
    except Exception:
        pass

    if os.getenv("PREDICT_FORCE_OK", "0") in {"1", "true", "TRUE"}:
        _queue_log("predict_force_ok", req_id=req_id)
        return {"status": "SUCCESS", "identifier": os.getenv("IDENTIFIER") or "default", "run_id": None, "predictions": []}

    service: Any | None = None
    try:
        service = _get_inferencer()
    except Exception:
        service = None

    prepared_df: Optional[pd.DataFrame] = None
    required_base_columns: List[str] = []
    prep_start = time.perf_counter()
    if req is not None and getattr(req, "data", None):
        try:
            prepared_df, required_base_columns = _prepare_dataframe_for_inference(req, service)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Failed to parse provided data: {exc}") from exc

    body_inference_length = getattr(req, "inference_length", None) if req else None
    effective_inference_length = inference_length if inference_length is not None else body_inference_length
    if effective_inference_length is None:
        effective_inference_length = 1

    df_for_inference = prepared_df
    if df_for_inference is None:
        if service is None:
            raise HTTPException(status_code=400, detail="No cached dataframe available and no data provided")
        # Use a deep copy of the cached dataframe to avoid concurrent mutation
        if hasattr(service, "get_df_copy"):
            df_for_inference = service.get_df_copy()
        else:
            df_for_inference = getattr(service, "df", None)
        if df_for_inference is None:
            raise HTTPException(status_code=400, detail="No cached dataframe available and no data provided")

    if required_base_columns:
        missing_base = [col for col in required_base_columns if col not in df_for_inference.columns]
        if missing_base:
            raise HTTPException(status_code=400, detail=f"Missing required feature columns: {', '.join(sorted(missing_base))}")

    prep_duration_ms = (time.perf_counter() - prep_start) * 1000.0
    prep_ms_int = int(prep_duration_ms)
    with queue_metrics_lock:
        queue_metrics["prep_samples"] += 1
        queue_metrics["total_prep_ms"] += prep_duration_ms
        queue_metrics["last_prep_ms"] = prep_ms_int
        if prep_ms_int > queue_metrics["max_prep_ms"]:
            queue_metrics["max_prep_ms"] = prep_ms_int

    with queue_metrics_lock:
        queue_metrics["enqueued"] += 1

    semaphore = _get_concurrency_semaphore()
    wait_started = time.perf_counter()
    await semaphore.acquire()
    wait_ms = (time.perf_counter() - wait_started) * 1000.0
    with queue_metrics_lock:
        queue_metrics["active"] += 1
        queue_metrics["wait_samples"] += 1
        queue_metrics["total_wait_ms"] += wait_ms
        wait_ms_int = int(wait_ms)
        queue_metrics["last_wait_ms"] = wait_ms_int
        if wait_ms_int > queue_metrics["max_wait_ms"]:
            queue_metrics["max_wait_ms"] = wait_ms_int
        active_now = queue_metrics["active"]
    _queue_log(
        "predict_inline_start",
        req_id=req_id,
        rows=len(df_for_inference.index) if isinstance(df_for_inference, pd.DataFrame) else None,
        inference_length=int(effective_inference_length or 0),
        active_workers=active_now,
        concurrency_limit=PREDICT_MAX_CONCURRENCY,
        wait_ms=wait_ms_int,
        prep_ms=prep_ms_int,
    )

    try:
        maybe_delay = getattr(service, "simulate_delay_if_enabled", None)
        if callable(maybe_delay):
            await maybe_delay()
    except Exception:
        pass

    start_exec = time.time()
    duration_s: Optional[float] = None
    duration_ms: int = 0
    result_df: Optional[pd.DataFrame] = None
    try:
        if service is None:
            with queue_metrics_lock:
                queue_metrics["error_500_total"] += 1
                queue_metrics["last_error_type"] = "InferencerUnavailable"
            _queue_log("predict_inline_no_inferencer", req_id=req_id)
            raise HTTPException(status_code=503, detail="Inference service not ready")

        result_df = await asyncio.to_thread(
            service.perform_inference,
            df_for_inference,
            inference_length=effective_inference_length,
        )
        if result_df is None:
            with queue_metrics_lock:
                queue_metrics["error_500_total"] += 1
                queue_metrics["last_error_type"] = "InferenceSkipped"
            if _PROMETHEUS_AVAILABLE:
                try:
                    JOB_OUTCOME.labels("skipped").inc()
                except Exception:
                    pass
            _queue_log("predict_inline_skipped", req_id=req_id)
            raise HTTPException(status_code=500, detail="Inference skipped (see server logs)")

        duration_s = time.time() - start_exec
        duration_ms = int(duration_s * 1000)
        with queue_metrics_lock:
            queue_metrics["completed"] += 1
            queue_metrics["last_duration_ms"] = duration_ms
            queue_metrics["last_error_type"] = None
            queue_metrics["exec_samples"] += 1
            queue_metrics["total_exec_ms"] += duration_ms
            if duration_ms > queue_metrics["max_exec_ms"]:
                queue_metrics["max_exec_ms"] = duration_ms
    except HTTPException:
        raise
    except KeyError as exc:
        with queue_metrics_lock:
            queue_metrics["last_error_type"] = "MissingColumns"
        detail = _format_missing_columns_error(exc)
        if _PROMETHEUS_AVAILABLE:
            try:
                JOB_OUTCOME.labels("client_error").inc()
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=f"Missing required columns for inference: {detail}" if detail else "Missing required columns for inference") from exc
    except Exception as exc:  # noqa: BLE001
        with queue_metrics_lock:
            queue_metrics["error_500_total"] += 1
            queue_metrics["last_error_type"] = exc.__class__.__name__
        if _PROMETHEUS_AVAILABLE:
            try:
                JOB_OUTCOME.labels("exception").inc()
            except Exception:
                pass
        _queue_log("predict_inline_error", req_id=req_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Inference execution failed") from exc
    finally:
        with queue_metrics_lock:
            queue_metrics["active"] = max(0, queue_metrics["active"] - 1)
            active_remaining = queue_metrics["active"]
        semaphore.release()
        _refresh_prometheus_metrics(duration_s)
        _queue_log(
            "predict_inline_active_update",
            req_id=req_id,
            active_workers=active_remaining,
            concurrency_limit=PREDICT_MAX_CONCURRENCY,
        )

    assert duration_s is not None and result_df is not None

    cols = ["value"] if "value" in result_df.columns else result_df.columns.tolist()
    predictions: List[Dict[str, Any]] = []
    for ts, row in result_df[cols].iterrows():
        try:
            ts_serial = ts.isoformat()
        except Exception:
            ts_serial = str(ts)
        entry: Dict[str, Any] = {"ts": ts_serial}
        for col in cols:
            val = row[col]
            try:
                if val is None or (isinstance(val, float) and math.isnan(val)) or pd.isna(val):
                    entry[col] = None
                elif isinstance(val, (int, float)):
                    if isinstance(val, float) and math.isinf(val):
                        entry[col] = None
                    else:
                        entry[col] = float(val)
                else:
                    try:
                        entry[col] = float(val)
                    except (TypeError, ValueError):
                        entry[col] = val
            except Exception:
                entry[col] = None
        predictions.append(entry)

    identifier = os.getenv("IDENTIFIER") or "default"
    response_payload = {
        "status": "SUCCESS",
        "identifier": identifier,
        "run_id": getattr(service, "current_run_id", None) if service else None,
        "predictions": predictions,
        "req_id": req_id,
        "cached": False,
    }

    try:
        if hasattr(service, "set_last_prediction"):
            service.set_last_prediction(response_payload)
        else:
            setattr(service, "last_prediction_response", response_payload.copy())
    except Exception:
        pass

    if _PROMETHEUS_AVAILABLE:
        try:
            JOBS_PROCESSED.inc()
            JOB_OUTCOME.labels("success").inc()
        except Exception:
            pass
    _queue_log(
        "predict_inline_success",
        req_id=req_id,
        duration_ms=duration_ms,
        wait_ms=int(wait_ms) if wait_ms is not None else None,
        prep_ms=prep_ms_int,
        predictions=len(predictions),
    )
    return response_payload

@app.get("/queue_stats")
def queue_stats():
    snapshot = metrics()
    snapshot.pop("status", None)
    with queue_metrics_lock:
        snapshot.update(queue_metrics)
    return {"status": "ok", **snapshot}


# ---------------- Startup Readiness Gate (optional) -----------------
@app.on_event("startup")
async def _startup_event_nonblocking():  # pragma: no cover (startup side-effect)
    """Non-blocking startup handler.

    Starts lightweight runtime bits synchronously (resource logger and worker tasks)
    and then schedules a background coroutine that waits for a model and performs
    optional prewarm/fallback. This avoids blocking uvicorn's startup/accept loop
    while preserving the WAIT_FOR_MODEL behavior when desired.
    """
    global _startup_ready_ms
    # Start optional resource logger (non-blocking)
    try:
        _start_resource_logger_if_enabled()
    except Exception:
        pass

    try:
        loop = asyncio.get_running_loop()
        executor = _get_threadpool_executor()
        loop.set_default_executor(executor)
        _queue_log(
            "threadpool_configured",
            max_workers=getattr(executor, "_max_workers", None),
            concurrency_limit=PREDICT_MAX_CONCURRENCY,
        )
    except Exception as exc:
        _queue_log("threadpool_configure_failed", error=str(exc))

    # Ensure process pool is ready early so that once model arrives we can serve instantly.
    try:
        _ensure_process_pool_ready(_current_model_snapshot())
    except Exception:
        pass

    # Start Prometheus exporter if available
    try:
        if _PROMETHEUS_AVAILABLE:
            start_http_server(9091)
            print("Started synchronous inference service | Metrics -> :9091", flush=True)
        else:
            print("Started synchronous inference service | Metrics disabled (prometheus_client not installed)", flush=True)
    except Exception:
        pass

    try:
        await _start_queue_monitor()
    except Exception:
        _queue_log("queue_monitor_start_failed")

    try:
        _ensure_event_loop_monitor_started()
    except Exception:
        pass

    async def _background_startup():
        nonlocal_ready_ms = None
        if not WAIT_FOR_MODEL:
            # No gating requested: mark ready immediately
            _startup_ready_ms = int((time.time() - _startup_epoch) * 1000)
            _queue_log("startup_no_wait_model", ready_ms=_startup_ready_ms)
            return

        deadline = time.time() + MODEL_WAIT_TIMEOUT
        logged_first = False
        while time.time() < deadline:
            try:
                inf = _get_inferencer()
                if inf.current_model is not None:
                    _startup_ready_ms = int((time.time() - _startup_epoch) * 1000)
                    _queue_log("startup_model_ready", ready_ms=_startup_ready_ms)
                    try:
                        if _PROMETHEUS_AVAILABLE:
                            MODEL_READY.set(1)
                    except Exception:
                        pass
                    # Fire prewarm (don't block readiness longer than needed)
                    try:
                        await _prewarm_if_needed()
                    except Exception:
                        pass
                    return
            except Exception:
                pass
            if not logged_first:
                _queue_log("startup_waiting_for_model", timeout_sec=MODEL_WAIT_TIMEOUT)
                logged_first = True
            await asyncio.sleep(1.0)

        # Timeout elapsed and still no model; attempt a last-chance fallback load in thread
        try:
            loaded = await asyncio.to_thread(_fallback_load_latest_model, "timeout_fallback")
            if loaded:
                inf = _get_inferencer()
                _startup_ready_ms = int((time.time() - _startup_epoch) * 1000)
                _queue_log("startup_model_ready_fallback", ready_ms=_startup_ready_ms, run_id=getattr(inf, "current_run_id", None))
                try:
                    if _PROMETHEUS_AVAILABLE:
                        MODEL_READY.set(1)
                except Exception:
                    pass
                try:
                    await _prewarm_if_needed()
                except Exception:
                    pass
                return
        except Exception:
            pass

        _startup_ready_ms = int((time.time() - _startup_epoch) * 1000)
        _queue_log("startup_model_wait_timeout", waited_ms=_startup_ready_ms)

    # schedule background startup task and don't await it here
    try:
        # Optionally start the Kafka/runtime main loop inside this process.
        # By default the container CMD launches `python main.py &` which starts
        # runtime in a separate process; starting it here would duplicate work.
        # Control via env: INFERENCE_START_IN_APP=1 to enable starting runtime in-app
        try:
            if os.getenv("INFERENCE_START_IN_APP", "0").lower() in {"1", "true", "yes"}:
                import threading as _th
                def _start_main_runtime():
                    try:
                        from main import start_runtime_safe  # type: ignore
                        start_runtime_safe()
                    except Exception as _e:
                        _queue_log("runtime_start_thread_error", error=str(_e))
                _th.Thread(target=_start_main_runtime, daemon=True).start()
            else:
                _queue_log("runtime_start_skipped_in_app")
        except Exception:
            pass
        # Also schedule the async background startup (model wait + prewarm) so
        # model readiness is handled without blocking the server accept loop.
        asyncio.create_task(_background_startup())
    except Exception:
        # fall back to ensure we don't block startup if create_task fails
        pass


@app.on_event("shutdown")
async def _shutdown_event():  # pragma: no cover (shutdown side-effect)
    try:
        await _stop_queue_monitor()
    except Exception:
        pass
    global _event_loop_monitor_task
    if _event_loop_monitor_task is not None:
        _event_loop_monitor_task.cancel()
        try:
            await _event_loop_monitor_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        _event_loop_monitor_task = None
    global _threadpool_executor
    if _threadpool_executor is not None:
        try:
            _threadpool_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _threadpool_executor = None

if ENABLE_PUBLISH_API:
    class PublishRequest(BaseModel):
        bucket: str = Field(default_factory=lambda: os.getenv("PROCESSED_BUCKET", "processed-data"))
        object_key: str = Field(default_factory=lambda: os.getenv("TEST_OBJECT_KEY", "test_processed_data.parquet"))
        count: int = Field(default=1, ge=1, le=500000)
        ttl_ms: Optional[int] = Field(default=None, ge=1, le=86400000)
        key_prefix: Optional[str] = None
        identifier: Optional[str] = Field(default_factory=lambda: os.getenv("IDENTIFIER"))

    def _get_publish_producer():
        global _publish_producer
        if _publish_producer is None:
            try:
                from kafka import KafkaProducer  # type: ignore
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"Kafka client not available: {e}")
            bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
            if not bootstrap:
                raise HTTPException(status_code=500, detail="KAFKA_BOOTSTRAP_SERVERS not set")
            _publish_producer = KafkaProducer(
                bootstrap_servers=bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )
        return _publish_producer

    @app.post("/publish_inference_claims")
    async def publish_inference_claims(req: "PublishRequest" = Body(...)):
        """Test-only: publish N inference claim messages to Kafka (inference-data).

        Enabled only when ENABLE_PUBLISH_API=1. Produces simple claim-check JSON with bucket/object.
        Optional TTL: sets headers.deadline_ms to now+ttl_ms.
        """
        if not ENABLE_PUBLISH_API:
            raise HTTPException(status_code=404, detail="Endpoint disabled")
        prod = _get_publish_producer()
        now_ms = int(time.time() * 1000)
        headers = None
        if req.ttl_ms:
            deadline = str(now_ms + int(req.ttl_ms))
            # kafka-python expects header keys as str and values as bytes
            headers = [("deadline_ms", deadline.encode("utf-8"))]
        sent = 0
        for i in range(int(req.count)):
            key = None
            if req.key_prefix:
                key = f"{req.key_prefix}-{i}"
            payload = {
                "bucket": req.bucket,
                "object": req.object_key,
            }
            if req.identifier:
                payload["identifier"] = req.identifier
            try:
                prod.send(_publish_topic, value=payload, key=key, headers=headers)
                sent += 1
            except Exception as e:  # noqa: BLE001
                # Surface exception type to aid debugging
                _queue_log("publish_claim_error", error=f"{e.__class__.__name__}: {e!s}")
                raise HTTPException(status_code=500, detail=f"Publish error at i={i}: {e.__class__.__name__}: {e}")
        try:
            prod.flush(timeout=10)
        except Exception:
            pass
        _queue_log("publish_claims_ok", topic=_publish_topic, count=sent)
        return {"status": "ok", "published": sent, "topic": _publish_topic}


