"""FastAPI inference API with bounded async queue for overload protection.

Adds an in-process asyncio.Queue between request intake and model inference:
  - Bounded size (QUEUE_MAXSIZE) -> overflow returns 429 immediately.
  - Worker tasks (QUEUE_WORKERS) pull jobs sequentially / limited parallel.
  - Per-job inference timeout (INFERENCE_TIMEOUT) returns 504 if exceeded.
  - Preserves original /predict response schema & lazy model loading.
  - Provides /queue_stats for simple operational insight.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, Body, Query, Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os, uuid, math, traceback, json, time
import threading
import asyncio, time
import pandas as pd

app = FastAPI(title="Inference Synchronous API")

# --- Optional structured resource logging (env-gated via ENABLE_RESOURCE_LOGS) ---
def _start_resource_logger_if_enabled():
    try:
        if os.getenv("ENABLE_RESOURCE_LOGS", "false").lower() not in {"1", "true", "yes"}:
            return
        try:
            import psutil  # type: ignore
        except Exception:
            print({"service": "inference", "event": "resource_logger_psutil_missing"})
            return

        def _log_resource_usage():
            try:
                proc = psutil.Process()  # current process
                proc.cpu_percent(interval=0.0)  # prime
                while True:
                    cpu = proc.cpu_percent(interval=0.1)
                    rss_mb = proc.memory_info().rss / (1024 * 1024)
                    print({"service": "inference", "event": "resource_usage", "cpu_percent": cpu, "mem_mb": round(rss_mb, 1)})
                    time.sleep(10)
            except Exception as e:  # pragma: no cover
                print({"service": "inference", "event": "resource_logger_error", "error": str(e)})

        threading.Thread(target=_log_resource_usage, daemon=True, name="resource-logger").start()
    except Exception:
        pass

# ---------------- Queue / Worker & Readiness Config (env-driven) -----------------
QUEUE_MAXSIZE = int(os.getenv("QUEUE_MAXSIZE", "40"))  # raised default for smoother concurrency
QUEUE_WORKERS = max(1, int(os.getenv("QUEUE_WORKERS", "1")))
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "15"))
WAIT_FOR_MODEL = os.getenv("WAIT_FOR_MODEL", "0").lower() in {"1", "true", "yes"}
MODEL_WAIT_TIMEOUT = float(os.getenv("MODEL_WAIT_TIMEOUT", "60"))  # seconds
PREWARM_ENABLED = os.getenv("INFERENCE_PREWARM", "1").lower() in {"1", "true", "yes"}

# Optional: expose a test-only Kafka publish endpoint to enqueue inference claims
ENABLE_PUBLISH_API = os.getenv("ENABLE_PUBLISH_API", "0").lower() in {"1", "true", "yes"}
_publish_producer = None  # lazy-init if endpoint used
_publish_topic = os.getenv("PUBLISH_TOPIC", os.getenv("CONSUMER_TOPIC_0", "inference-data"))

_startup_epoch = time.time()
_startup_ready_ms: float | None = None


_inference_queue: asyncio.Queue | None = None
_workers_started = False

# Serialize underlying model.predict / perform_inference calls to avoid shared state races.
_MODEL_INFER_LOCK = threading.Lock()

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
    # error metrics
    "error_500_total": 0,
    "last_error_type": None,
}

# (Removed rolling inference duration tracking in rollback)

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

class _InferenceJob:
    __slots__ = ("req_id", "req_model", "inference_length_q", "future", "enqueue_time")
    def __init__(self, req_id: str, req_model: PredictRequest | None, inference_length_q: int | None):
        self.req_id = req_id
        self.req_model = req_model
        self.inference_length_q = inference_length_q
        loop = asyncio.get_event_loop()
        self.future: asyncio.Future = loop.create_future()
        self.enqueue_time = time.time()

def _queue_log(event: str, **extra):  # central helper for structured logs
    try:
        payload = {"service": "inference", "event": event}
        payload.update(extra)
        print(payload, flush=True)
    except Exception:
        pass

async def _start_workers_once():
    global _workers_started, _inference_queue
    if _workers_started:
        return
    _inference_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    for i in range(QUEUE_WORKERS):
        asyncio.create_task(_worker_loop(i))
        # explicit per-worker start log (human readable + structured)
        print(f"[queue] worker-{i} started", flush=True)
        _queue_log("queue_worker_started", worker=i)
    _workers_started = True
    _queue_log("queue_workers_started", workers=QUEUE_WORKERS, maxsize=QUEUE_MAXSIZE, timeout=INFERENCE_TIMEOUT)

async def _worker_loop(worker_idx: int):  # pragma: no cover (long-lived)
    while True:
        job: _InferenceJob = await _inference_queue.get()  # type: ignore
        queue_metrics["active"] += 1
        wait_ms = int((time.time() - job.enqueue_time) * 1000)
        queue_metrics["total_wait_ms"] += wait_ms
        queue_metrics["wait_samples"] += 1
        _queue_log("queue_job_start", req_id=job.req_id, worker=worker_idx, active=queue_metrics["active"], qsize=_inference_queue.qsize() if _inference_queue else None, waited_ms=wait_ms)
        try:
            result = await asyncio.wait_for(_execute_inference(job.req_model, job.inference_length_q, job.req_id), timeout=INFERENCE_TIMEOUT)
            if not job.future.done():
                job.future.set_result(result)
            queue_metrics["completed"] += 1
            _queue_log("queue_job_done", req_id=job.req_id, worker=worker_idx, active=queue_metrics["active"], completed=queue_metrics["completed"])
        except asyncio.TimeoutError:
            queue_metrics["timeouts"] += 1
            if not job.future.done():
                job.future.set_exception(HTTPException(status_code=504, detail="Inference timed out"))
            _queue_log("queue_job_timeout", req_id=job.req_id, worker=worker_idx, timeouts=queue_metrics["timeouts"])
        except HTTPException as he:
            if he.status_code >= 500:
                queue_metrics["error_500_total"] += 1
                queue_metrics["last_error_type"] = "HTTPException"
            if not job.future.done():
                job.future.set_exception(he)
            _queue_log("queue_job_http_error", req_id=job.req_id, status_code=he.status_code)
        except Exception as e:  # noqa: BLE001
            queue_metrics["error_500_total"] += 1
            queue_metrics["last_error_type"] = e.__class__.__name__
            if not job.future.done():
                job.future.set_exception(HTTPException(status_code=500, detail=f"Worker error: {e}"))
            _queue_log("queue_job_error", req_id=job.req_id, error=str(e))
        finally:
            queue_metrics["active"] -= 1
            try:
                _inference_queue.task_done()  # type: ignore
            except Exception:
                pass

async def _execute_inference(req: PredictRequest | None, inference_length_param: int | None, req_id: str):
    inf = _get_inferencer()
    # Attempt autoload if model not present
    if inf.current_model is None:
        try:  # pragma: no cover
            import importlib
            main_mod = importlib.import_module("main")  # type: ignore
            if hasattr(main_mod, "_attempt_load_promoted"):
                main_mod._attempt_load_promoted(inf)
        except Exception:  # noqa: BLE001
            pass
    if inf.current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Data resolution
    if req and req.data:
        try:
            df_tmp = pd.DataFrame(req.data)
            if req.index_col and req.index_col in df_tmp.columns:
                df_tmp[req.index_col] = pd.to_datetime(df_tmp[req.index_col], errors="coerce")
                df_tmp = df_tmp.set_index(req.index_col).sort_index()
            elif df_tmp.columns[0].lower() in {"ts", "time", "timestamp"}:
                idx_col = df_tmp.columns[0]
                df_tmp[idx_col] = pd.to_datetime(df_tmp[idx_col], errors="coerce")
                df_tmp = df_tmp.set_index(idx_col).sort_index()
            else:
                raise ValueError("No recognizable index column; supply 'index_col'.")
            df = df_tmp
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Failed to parse provided data: {e}")
    else:
        if inf.df is None:
            raise HTTPException(status_code=400, detail="No cached dataframe available and no data provided.")
        df = inf.df

    # Underlying busy guard (still allow cached) — primarily defensive if model sets busy flag internally
    if getattr(inf, 'busy', False):
        allow_cached = os.getenv("PREDICT_ALLOW_CACHED", "1").lower() in {"1","true","yes"}
        if allow_cached and hasattr(inf, 'last_prediction_response') and inf.last_prediction_response:
            cached = inf.last_prediction_response.copy()
            cached["status"] = "SUCCESS_CACHED"
            cached["cached"] = True
            cached["req_id"] = req_id
            queue_metrics["served_cached"] += 1
            _queue_log("predict_served_cached_busy", req_id=req_id, served_cached=queue_metrics["served_cached"])
            return cached
        queue_metrics["rejected_busy"] += 1
        _queue_log("predict_rejected_busy", req_id=req_id, rejected_busy=queue_metrics["rejected_busy"])
        raise HTTPException(status_code=429, detail="Inference busy, try again")

    eff_len = inference_length_param if inference_length_param is not None else (req.inference_length if req and req.inference_length is not None else 1)
    if os.getenv("PREDICT_STUB", "0") in {"1", "true", "TRUE"}:
        return {"status": "SUCCESS", "identifier": os.getenv("IDENTIFIER") or "default", "run_id": getattr(inf, "current_run_id", None), "predictions": []}

    try:
        # Wrap perform_inference in a thread function that enforces single-flight execution.
        def _locked_infer():
            with _MODEL_INFER_LOCK:
                return inf.perform_inference(df, inference_length=eff_len)
        preds = await asyncio.to_thread(_locked_infer)
        if preds is None:
            raise HTTPException(status_code=500, detail="Inference skipped (see server logs)")
        identifier = (os.getenv("IDENTIFIER") or "default") or "default"
        cols = [c for c in (["value"] if "value" in preds.columns else preds.columns.tolist())]
        pred_list = []
        for ts, row in preds[cols].iterrows():
            try:
                ts_serial = ts.isoformat()
            except Exception:
                ts_serial = str(ts)
            entry = {"ts": ts_serial}
            for c in cols:
                try:
                    val = row[c]
                    if val is None or (isinstance(val, (float, int)) and (not math.isfinite(float(val)))):
                        entry[c] = None
                    elif pd.isna(val):  # type: ignore[attr-defined]
                        entry[c] = None
                    else:
                        entry[c] = float(val)
                except Exception:
                    entry[c] = None
            pred_list.append(entry)
        resp = {
            "status": "SUCCESS",
            "identifier": identifier,
            "run_id": getattr(inf, "current_run_id", None),
            "predictions": pred_list,
        }
        try:
            inf.last_prediction_response = resp  # type: ignore[attr-defined]
        except Exception:
            pass
        return resp
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        if any(k in str(e).lower() for k in ["broken pipe", "connection reset", "client disconnected"]):
            raise HTTPException(status_code=499, detail="Client connection lost during response")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

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
    return {"status": "ok", "service": "inference-api", "model_ready": model_ready, "startup_ready_ms": _startup_ready_ms}

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
    qsize = _inference_queue.qsize() if _inference_queue else 0
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
    await _start_workers_once()
    req_id = uuid.uuid4().hex[:8]
    # Serve cached response instantly for empty {} request if available (no new job enqueued)
    try:
        if (req is None) or ( (not getattr(req, 'data', None)) and inference_length is None and getattr(req, 'inference_length', None) is None):
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

    job = _InferenceJob(req_id=req_id, req_model=req, inference_length_q=inference_length)
    if _inference_queue is None:
        raise HTTPException(status_code=500, detail="Queue not initialized")
    try:
        _inference_queue.put_nowait(job)
        queue_metrics["enqueued"] += 1
        _queue_log("queue_job_enqueued", req_id=req_id, qsize=_inference_queue.qsize(), enqueued=queue_metrics["enqueued"], active=queue_metrics["active"])
    except asyncio.QueueFull:
        queue_metrics["rejected_full"] += 1
        _queue_log("queue_job_rejected_full", req_id=req_id, rejected=queue_metrics["rejected_full"], qsize=_inference_queue.qsize() if _inference_queue else None)
        # Provide Retry-After hint (100ms) to cooperative clients
        raise HTTPException(status_code=429, detail="Server busy, try again", headers={"Retry-After": "0.1"})

    try:
        return await job.future
    except HTTPException as he:  # Mask internal 5xx details with generic message
        if he.status_code >= 500:
            queue_metrics["error_500_total"] += 1
            queue_metrics["last_error_type"] = "HTTPException"
            _queue_log("predict_internal_error_masked", req_id=req_id, original_detail=str(he.detail))
            raise HTTPException(status_code=500, detail="Internal inference error")
        raise
    except Exception as e:  # noqa: BLE001
        queue_metrics["error_500_total"] += 1
        queue_metrics["last_error_type"] = e.__class__.__name__
        _queue_log("predict_unexpected_error", req_id=req_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal inference error")

@app.get("/queue_stats")
def queue_stats():
    qsize = _inference_queue.qsize() if _inference_queue else 0
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

    # Ensure workers are ready early so that once model arrives we can serve instantly.
    try:
        await _start_workers_once()
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

