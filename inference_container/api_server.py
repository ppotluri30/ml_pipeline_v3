"""FastAPI inference API fronting a process pool for bounded parallel inference."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from data_utils import strip_timezones, time_to_feature

try:  # Prefer package-style import when available (local tests)
    from inference_container.process_pool import (
        InferenceHTTPError,
        build_job_payload,
        ensure_process_pool,
        pending_jobs,
        release_slot,
        reinitialize_process_pool,
        submit_inference_job,
        try_acquire_slot,
    )
except ModuleNotFoundError:  # Fallback for in-container execution
    from process_pool import (  # type: ignore
        InferenceHTTPError,
        build_job_payload,
        ensure_process_pool,
        pending_jobs,
        release_slot,
        reinitialize_process_pool,
        submit_inference_job,
        try_acquire_slot,
    )

# Prometheus metrics (optional dependency)
try:
    from prometheus_client import Gauge, Counter, Histogram, start_http_server  # type: ignore
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


POOL_SCALE_MIN = int(os.getenv("POOL_SCALE_MIN", "1"))
POOL_SCALE_MAX = int(os.getenv("POOL_SCALE_MAX", "64"))

# Worker and queue sizing defaults
DEFAULT_QUEUE_WORKERS = int(os.getenv("QUEUE_WORKERS", "4"))
QUEUE_WORKERS = max(POOL_SCALE_MIN, min(POOL_SCALE_MAX, DEFAULT_QUEUE_WORKERS))
QUEUE_MAXSIZE = max(1, int(os.getenv("QUEUE_MAXSIZE", "32")))
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "30"))
QUEUE_MONITOR_INTERVAL_SECS = float(os.getenv("QUEUE_MONITOR_INTERVAL_SECS", "0.5"))

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
    global QUEUE_WORKERS
    try:
        desired = int(payload.get("workers", 0))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid 'workers' value")
    if desired < POOL_SCALE_MIN or desired > POOL_SCALE_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"workers must be between {POOL_SCALE_MIN} and {POOL_SCALE_MAX}",
        )

    old = QUEUE_WORKERS
    if desired == old:
        return {"old_workers": old, "new_workers": desired, "status": "no_change"}

    QUEUE_WORKERS = desired
    reinitialize_process_pool(desired, QUEUE_MAXSIZE, _current_model_snapshot())
    _refresh_prometheus_metrics()
    _queue_log("queue_workers_scaled", old=old, new=desired, reason="manual_process_pool")
    print(
        f"Reinitialized process pool from {old} -> {desired} workers (manual request via /scale_workers)",
        flush=True,
    )
    return {"old_workers": old, "new_workers": desired, "status": "pool_reinitialized"}
ENABLE_PUBLISH_API = os.getenv("ENABLE_PUBLISH_API", "0").lower() in {"1", "true", "yes"}
_publish_producer = None  # lazy-init if endpoint used
_publish_topic = os.getenv("PUBLISH_TOPIC", os.getenv("CONSUMER_TOPIC_0", "inference-data"))

_startup_epoch = time.time()
_startup_ready_ms: float | None = None


_pool_started = False
_queue_monitor_task: asyncio.Task | None = None
_queue_monitor_stop_event: asyncio.Event | None = None

queue_metrics = {
    "enqueued": 0,
    "rejected_full": 0,
    "rejected_busy": 0,
    "active": 0,
    "completed": 0,
    "timeouts": 0,
    "served_cached": 0,
    # wait time accounting
    "total_wait_ms": 0,
    "wait_samples": 0,
    "last_wait_ms": 0,
    "last_duration_ms": 0,
    "last_worker_id": None,
    # error metrics
    "error_500_total": 0,
    "last_error_type": None,
}

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
    try:
        return pending_jobs()
    except Exception:
        return 0


async def _start_queue_monitor():
    global _queue_monitor_task, _queue_monitor_stop_event
    if not _PROMETHEUS_AVAILABLE:
        return
    if _queue_monitor_task and not _queue_monitor_task.done():
        return
    stop_event = asyncio.Event()
    _queue_monitor_stop_event = stop_event

    async def _monitor_loop():
        _queue_log("queue_monitor_started", interval=QUEUE_MONITOR_INTERVAL_SECS)
        try:
            while not stop_event.is_set():
                try:
                    qsize = _safe_queue_size()
                    QUEUE_LEN.set(qsize)
                    QUEUE_OLDEST_WAIT.set(0.0)
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=QUEUE_MONITOR_INTERVAL_SECS)
                except asyncio.TimeoutError:
                    continue
        finally:
            try:
                qsize = _safe_queue_size()
                QUEUE_LEN.set(qsize)
                QUEUE_OLDEST_WAIT.set(0.0)
            except Exception:
                pass
            _queue_log("queue_monitor_stopped")

    _queue_monitor_task = asyncio.create_task(_monitor_loop())


async def _stop_queue_monitor():
    global _queue_monitor_task, _queue_monitor_stop_event
    if _queue_monitor_stop_event:
        _queue_monitor_stop_event.set()
    task = _queue_monitor_task
    if task:
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            with contextlib.suppress(Exception):
                await task
        finally:
            _queue_monitor_task = None
            _queue_monitor_stop_event = None
    else:
        _queue_monitor_stop_event = None

def _request_to_dict(req: PredictRequest | None) -> Optional[Dict[str, Any]]:
    if req is None:
        return None
    if hasattr(req, "model_dump"):
        return req.model_dump()  # type: ignore[attr-defined]
    if hasattr(req, "dict"):
        return req.dict()
    try:
        return json.loads(req.json())
    except Exception:
        return None


TIME_FEATURE_COLUMNS = {
    "min_of_day_sin",
    "min_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
}


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
                columns.update(list(extra))
            except Exception:
                pass
    return {col for col in columns if col not in TIME_FEATURE_COLUMNS}


def _prepare_dataframe_for_inference(req_obj: PredictRequest, service: Any | None) -> tuple[pd.DataFrame, List[str]]:
    """Validate the inbound payload and produce a feature-engineered DataFrame.

    Returns the prepared dataframe plus the list of required base feature columns enforced.
    Raises HTTPException(400) for validation failures so the caller can short-circuit.
    """

    data = getattr(req_obj, "data", None)
    if not data:
        raise HTTPException(status_code=400, detail="Request payload must include a non-empty 'data' object")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="'data' must be an object mapping column names to arrays")

    try:
        df_tmp = pd.DataFrame(data)
    except Exception as exc:  # noqa: BLE001
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
            idx = pd.to_datetime(df_tmp[candidate], errors="coerce")
        except Exception:
            continue
        if idx.isna().any():
            raise HTTPException(status_code=400, detail=f"Column '{candidate}' contains invalid timestamps")
        df_tmp = df_tmp.drop(columns=[candidate])
        df_tmp.index = idx
        assigned_index = True
        break

    if not assigned_index or not isinstance(df_tmp.index, pd.DatetimeIndex):
        raise HTTPException(status_code=400, detail="Unable to determine a valid datetime index from request data")

    df_tmp = df_tmp.sort_index()
    df_tmp, _ = strip_timezones(df_tmp)
    if not isinstance(df_tmp.index, pd.DatetimeIndex):
        raise HTTPException(status_code=400, detail="Index must be datetime after timezone normalization")
    if df_tmp.empty:
        raise HTTPException(status_code=400, detail="Request data must include at least one row after normalization")

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


def _refresh_prometheus_metrics(active_now: Optional[int] = None, duration_s: Optional[float] = None, wait_seconds: Optional[float] = None) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    try:
        if active_now is None:
            active_now = max(0, queue_metrics.get("active", 0))
        WORKERS_TOTAL.set(QUEUE_WORKERS)
        ACTIVE_WORKERS.set(active_now)
        WORKERS_BUSY.set(active_now)
        WORKERS_IDLE.set(max(0, QUEUE_WORKERS - active_now))
        if QUEUE_WORKERS:
            WORKER_UTILIZATION.set(min(1.0, active_now / QUEUE_WORKERS))
        qsize = pending_jobs()
        QUEUE_LEN.set(qsize)
        if wait_seconds is not None:
            QUEUE_WAIT_LATEST.set(wait_seconds)
            QUEUE_WAIT_TIME.observe(wait_seconds)
        if duration_s is not None:
            INFERENCE_LATENCY.observe(max(0.0, duration_s))
            INFERENCE_DURATION_LATEST.set(max(0.0, duration_s))
    except Exception:
        pass


def _ensure_process_pool_ready(snapshot: Dict[str, Any]) -> None:
    global _pool_started
    ensure_process_pool(QUEUE_WORKERS, QUEUE_MAXSIZE, snapshot)
    _pool_started = True
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
                await asyncio.to_thread(inf.perform_inference, inf.df, 1)
            except Exception as ie:  # noqa: BLE001
                _queue_log("prewarm_fail", error=str(ie))
        # Prophet / statsforecast usually compile lazily; trigger generic perform_inference
        elif getattr(inf, "model_class", "").lower() in {"prophet", "statsforecast"} and inf.df is not None:
            try:
                await asyncio.to_thread(inf.perform_inference, inf.df, 1)
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
        return {
            "status": "ok",
            "model_loaded": inf.current_model is not None,
            "has_df": inf.df is not None,
            "busy": getattr(inf, "busy", False),
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
    qsize = _safe_queue_size()
    avg_wait = None
    if queue_metrics["wait_samples"]:
        avg_wait = queue_metrics["total_wait_ms"] / max(1, queue_metrics["wait_samples"])
    return {
        "queue_length": qsize,
        "workers": QUEUE_WORKERS,
        "completed": queue_metrics["completed"],
        "rejected": queue_metrics["rejected_full"] + queue_metrics["rejected_busy"],
        "timeouts": queue_metrics["timeouts"],
        "error_500_total": queue_metrics["error_500_total"],
        "last_error_type": queue_metrics["last_error_type"],
        "model_loaded": bool(getattr(inf, "current_model", None)) if inf else False,
        "current_model_hash": getattr(inf, "current_config_hash", None) if inf else None,
        "current_run_id": getattr(inf, "current_run_id", None) if inf else None,
        "current_model_type": getattr(inf, "model_type", None) if inf else None,
        "startup_latency_ms": _startup_ready_ms,
        "prewarm_latency_ms": getattr(inf, "last_prewarm_ms", None) if inf else None,
        "average_queue_wait_ms": avg_wait,
        "last_queue_wait_ms": queue_metrics["last_wait_ms"],
        "last_inference_duration_ms": queue_metrics["last_duration_ms"],
        "served_cached": queue_metrics["served_cached"],
        "max_queue": QUEUE_MAXSIZE,
        "build_version": build_version,
        "status": "ok",
    }

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
    # Serve cached response instantly for empty {} request if available (no new job enqueued)
    try:
        if _cache_enabled() and ((req is None) or ((not getattr(req, 'data', None)) and inference_length is None and getattr(req, 'inference_length', None) is None)):
            inf_cached = _get_inferencer()
            if hasattr(inf_cached, 'last_prediction_response') and getattr(inf_cached, 'last_prediction_response'):
                cached = getattr(inf_cached, 'last_prediction_response').copy()
                cached["status"] = "SUCCESS_CACHED"
                cached["cached"] = True
                cached["req_id"] = req_id
                queue_metrics["served_cached"] += 1
                _queue_log("predict_served_cached_direct", req_id=req_id, served_cached=queue_metrics["served_cached"])
                return cached
    except Exception:
        # Silent failover to normal path
        pass
    if os.getenv("PREDICT_FORCE_OK", "0") in {"1", "true", "TRUE"}:
        _queue_log("predict_force_ok", req_id=req_id)
        return {"status": "SUCCESS", "identifier": os.getenv("IDENTIFIER") or "default", "run_id": None, "predictions": []}

    prepared_df = None
    required_base_columns: List[str] = []
    if req is not None and getattr(req, "data", None):
        service = None
        try:
            service = _get_inferencer()
        except Exception:
            service = None
        try:
            prepared_df, required_base_columns = _prepare_dataframe_for_inference(req, service)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Failed to parse provided data: {exc}") from exc

    snapshot = _current_model_snapshot()
    _ensure_process_pool_ready(snapshot)

    if not try_acquire_slot():
        queue_metrics["rejected_full"] += 1
        _queue_log("queue_job_rejected_full", req_id=req_id, rejected=queue_metrics["rejected_full"], qsize=_safe_queue_size())
        if _PROMETHEUS_AVAILABLE:
            try:
                JOB_OUTCOME.labels("queue_full").inc()
            except Exception:
                pass
        raise HTTPException(status_code=429, detail="Server busy, try again", headers={"Retry-After": "0.1"})

    body_inference_length = getattr(req, "inference_length", None) if req else None
    effective_inference_length = inference_length if inference_length is not None else body_inference_length

    enqueue_time = time.time()
    queue_metrics["enqueued"] += 1
    queue_metrics["last_wait_ms"] = 0
    queue_metrics["wait_samples"] += 1
    payload = build_job_payload(
        prepared_df,
        effective_inference_length,
        req_id,
        snapshot,
        expected_base_columns=required_base_columns,
    )
    payload["queue_workers"] = QUEUE_WORKERS
    _queue_log("queue_job_enqueued", req_id=req_id, qsize=_safe_queue_size(), enqueued=queue_metrics["enqueued"], active=queue_metrics["active"])

    try:
        future = submit_inference_job(payload)
    except Exception as exc:  # noqa: BLE001
        release_slot()
        queue_metrics["rejected_full"] += 1
        queue_metrics["last_error_type"] = exc.__class__.__name__
        _queue_log("queue_job_submit_error", req_id=req_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal inference error")

    queue_metrics["active"] += 1
    queue_metrics["last_worker_id"] = None
    _refresh_prometheus_metrics(active_now=queue_metrics["active"], wait_seconds=0.0)
    _queue_log("queue_job_start", req_id=req_id, active=queue_metrics["active"], qsize=_safe_queue_size(), waited_ms=0)

    start_exec = time.time()
    wrapped_future = asyncio.wrap_future(future)
    try:
        worker_payload = await asyncio.wait_for(wrapped_future, timeout=INFERENCE_TIMEOUT)
        end_time = time.time()
        duration_s = end_time - start_exec
        total_elapsed = end_time - enqueue_time
        wait_seconds = max(0.0, total_elapsed - duration_s)
        worker_response: Any = worker_payload
        worker_meta: Dict[str, Any] = {}
        if isinstance(worker_payload, dict):
            maybe_response = worker_payload.get("response")
            maybe_meta = worker_payload.get("meta")
            if maybe_response is not None and isinstance(maybe_response, dict):
                worker_response = maybe_response
            if isinstance(maybe_meta, dict):
                worker_meta = maybe_meta
        worker_id = worker_meta.get("worker_id") if isinstance(worker_meta, dict) else None
        worker_duration_ms = worker_meta.get("duration_ms") if isinstance(worker_meta, dict) else None
        worker_predictions = worker_meta.get("predictions") if isinstance(worker_meta, dict) else None
        if worker_id is not None:
            queue_metrics["last_worker_id"] = worker_id
        queue_metrics["completed"] += 1
        queue_metrics["last_duration_ms"] = int(duration_s * 1000)
        queue_metrics["last_wait_ms"] = int(wait_seconds * 1000)
        queue_metrics["total_wait_ms"] += queue_metrics["last_wait_ms"]
        if _PROMETHEUS_AVAILABLE:
            try:
                JOBS_PROCESSED.inc()
                JOB_OUTCOME.labels("success").inc()
            except Exception:
                pass
        _refresh_prometheus_metrics(active_now=queue_metrics["active"], duration_s=duration_s, wait_seconds=wait_seconds)
        _queue_log(
            "queue_job_done",
            req_id=req_id,
            active=queue_metrics["active"],
            completed=queue_metrics["completed"],
            wait_ms=queue_metrics["last_wait_ms"],
            duration_ms=queue_metrics["last_duration_ms"],
            worker_id=worker_id,
            worker_duration_ms=worker_duration_ms,
            worker_predictions=worker_predictions,
        )
        return worker_response
    except asyncio.TimeoutError:
        future.cancel()
        end_time = time.time()
        duration_s = end_time - start_exec
        total_elapsed = end_time - enqueue_time
        wait_seconds = max(0.0, total_elapsed - duration_s)
        queue_metrics["timeouts"] += 1
        queue_metrics["last_error_type"] = "TimeoutError"
        queue_metrics["last_duration_ms"] = int(duration_s * 1000)
        queue_metrics["last_wait_ms"] = int(wait_seconds * 1000)
        queue_metrics["last_worker_id"] = None
        if _PROMETHEUS_AVAILABLE:
            try:
                JOB_OUTCOME.labels("timeout").inc()
            except Exception:
                pass
        _refresh_prometheus_metrics(active_now=queue_metrics["active"], duration_s=duration_s, wait_seconds=wait_seconds)
        _queue_log(
            "queue_job_timeout",
            req_id=req_id,
            timeout=INFERENCE_TIMEOUT,
            wait_ms=queue_metrics["last_wait_ms"],
            duration_ms=queue_metrics["last_duration_ms"],
        )
        raise HTTPException(status_code=504, detail="Inference timed out")
    except InferenceHTTPError as ihe:
        end_time = time.time()
        duration_s = end_time - start_exec
        total_elapsed = end_time - enqueue_time
        wait_seconds = max(0.0, total_elapsed - duration_s)
        queue_metrics["last_duration_ms"] = int(duration_s * 1000)
        queue_metrics["last_wait_ms"] = int(wait_seconds * 1000)
        worker_id = getattr(ihe, "worker_id", None)
        if worker_id is not None:
            queue_metrics["last_worker_id"] = worker_id
        else:
            queue_metrics["last_worker_id"] = None
        if ihe.status_code >= 500:
            queue_metrics["error_500_total"] += 1
            queue_metrics["last_error_type"] = "InferenceHTTPError"
            if _PROMETHEUS_AVAILABLE:
                try:
                    JOB_OUTCOME.labels("server_error").inc()
                except Exception:
                    pass
        elif ihe.status_code == 429:
            queue_metrics["rejected_busy"] += 1
            if _PROMETHEUS_AVAILABLE:
                try:
                    JOB_OUTCOME.labels("busy").inc()
                except Exception:
                    pass
        else:
            if _PROMETHEUS_AVAILABLE:
                try:
                    JOB_OUTCOME.labels("client_error").inc()
                except Exception:
                    pass
        _refresh_prometheus_metrics(active_now=queue_metrics["active"], duration_s=duration_s, wait_seconds=wait_seconds)
        _queue_log("queue_job_http_error", req_id=req_id, status_code=ihe.status_code, worker_id=worker_id)
        raise HTTPException(status_code=ihe.status_code, detail=ihe.detail)
    except Exception as exc:  # noqa: BLE001
        end_time = time.time()
        duration_s = end_time - start_exec
        total_elapsed = end_time - enqueue_time
        wait_seconds = max(0.0, total_elapsed - duration_s)
        queue_metrics["last_duration_ms"] = int(duration_s * 1000)
        queue_metrics["last_wait_ms"] = int(wait_seconds * 1000)
        queue_metrics["error_500_total"] += 1
        queue_metrics["last_error_type"] = exc.__class__.__name__
        if _PROMETHEUS_AVAILABLE:
            try:
                JOB_OUTCOME.labels("exception").inc()
            except Exception:
                pass
        _refresh_prometheus_metrics(active_now=queue_metrics["active"], duration_s=duration_s, wait_seconds=wait_seconds)
        worker_id = getattr(exc, "worker_id", None)
        if worker_id is not None:
            queue_metrics["last_worker_id"] = worker_id
        else:
            queue_metrics["last_worker_id"] = None
        _queue_log("queue_job_error", req_id=req_id, error=str(exc), worker_id=worker_id)
        raise HTTPException(status_code=500, detail="Internal inference error")
    finally:
        queue_metrics["active"] = max(0, queue_metrics["active"] - 1)
        release_slot()
        _refresh_prometheus_metrics(active_now=queue_metrics["active"])

@app.get("/queue_stats")
def queue_stats():
    qsize = _safe_queue_size()
    return {"status": "ok", "qsize": qsize, **queue_metrics, "workers": QUEUE_WORKERS, "maxsize": QUEUE_MAXSIZE}


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

    # Ensure process pool is ready early so that once model arrives we can serve instantly.
    try:
        _ensure_process_pool_ready(_current_model_snapshot())
    except Exception:
        pass

    # Start Prometheus exporter if available
    try:
        if _PROMETHEUS_AVAILABLE:
            start_http_server(9091)
            print(f"Started process pool with {QUEUE_WORKERS} workers | Queue maxsize = {QUEUE_MAXSIZE} | Metrics -> :9091", flush=True)
        else:
            print(f"Started process pool with {QUEUE_WORKERS} workers | Queue maxsize = {QUEUE_MAXSIZE} | Metrics disabled (prometheus_client not installed)", flush=True)
    except Exception:
        pass

    try:
        await _start_queue_monitor()
    except Exception:
        _queue_log("queue_monitor_start_failed")

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


