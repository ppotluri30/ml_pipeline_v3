"""Process-based inference execution for FastAPI /predict endpoint."""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import math
import multiprocessing as mp
import os
import sys
import threading
import time
import types
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:  # Python 3.8+ provides BrokenProcessPool
    from concurrent.futures import BrokenProcessPool
except ImportError:  # pragma: no cover
    class BrokenProcessPool(Exception):
        """Compatibility shim when BrokenProcessPool is unavailable."""


__all__ = [
    "InferenceHTTPError",
    "ensure_process_pool",
    "reinitialize_process_pool",
    "try_acquire_slot",
    "release_slot",
    "pending_jobs",
    "submit_inference_job",
    "build_job_payload",
]


class InferenceHTTPError(Exception):
    def __init__(self, status_code: int = 500, detail: str = "", worker_id: Optional[int] = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.worker_id = worker_id

    def __reduce__(self):  # pragma: no cover - exercised implicitly in multiprocess pickling
        return (self.__class__, (self.status_code, self.detail, self.worker_id))


_pool: Optional[ProcessPoolExecutor] = None
_pool_lock = threading.Lock()
_queue_maxsize = 0
_queue_workers = 1
_pending_jobs = 0
_pending_lock = threading.Lock()
_last_snapshot: Dict[str, Any] | None = None
_mp_context: Optional[mp.context.BaseContext] = None


def _worker_log(event: str, req_id: Optional[str] = None, **fields: Any) -> None:
    """Emit a structured log from within a worker process with worker attribution."""

    try:
        payload: Dict[str, Any] = {
            "service": "inference",
            "event": event,
            "worker_id": os.getpid(),
            "source": "worker",
        }
        if req_id is not None:
            payload["req_id"] = req_id
        payload.update(fields)
        print(payload, flush=True)
    except Exception:
        pass


def _resolve_mp_context() -> Optional[mp.context.BaseContext]:
    global _mp_context
    if _mp_context is not None:
        return _mp_context
    method = os.getenv("PROCESS_POOL_START_METHOD", "spawn").strip().lower()
    if method in {"spawn", "fork", "forkserver"}:
        try:
            _mp_context = mp.get_context(method)
        except ValueError:
            _mp_context = None
    else:
        _mp_context = None
    if _mp_context is None:
        if method:
            try:
                print({"service": "inference", "event": "process_pool_context_fallback", "requested": method}, flush=True)
            except Exception:
                pass
    else:
        try:
            resolved = getattr(_mp_context, "get_start_method", lambda: method)()
            print({
                "service": "inference",
                "event": "process_pool_context_selected",
                "requested": method or None,
                "resolved": resolved,
            }, flush=True)
        except Exception:
            pass
    return _mp_context


def ensure_process_pool(queue_workers: int, queue_maxsize: int, snapshot: Dict[str, Any]) -> None:
    global _pool, _queue_maxsize, _queue_workers, _last_snapshot
    _queue_maxsize = max(1, int(queue_maxsize))
    _queue_workers = max(1, int(queue_workers))
    _last_snapshot = snapshot
    if _pool is not None:
        return
    with _pool_lock:
        if _pool is None:
            executor_kwargs = {
                "max_workers": _queue_workers,
                "initializer": _worker_initializer,
                "initargs": (snapshot,),
            }
            context = _resolve_mp_context()
            if context is not None:
                executor_kwargs["mp_context"] = context
            _pool = ProcessPoolExecutor(**executor_kwargs)


def reinitialize_process_pool(queue_workers: int, queue_maxsize: int, snapshot: Dict[str, Any]) -> None:
    """Tear down the existing process pool and recreate it with new settings."""

    # Normalize and store the latest desired settings up-front.
    queue_workers = max(1, int(queue_workers))
    queue_maxsize = max(1, int(queue_maxsize))

    global _queue_workers, _queue_maxsize, _last_snapshot
    _queue_workers = queue_workers
    _queue_maxsize = queue_maxsize
    _last_snapshot = snapshot

    # Reset accounting for pending slots; cancelling futures will drop active jobs.
    with _pending_lock:
        global _pending_jobs
        _pending_jobs = 0

    # Shutdown the pool and create a fresh executor.
    _reset_pool()
    ensure_process_pool(_queue_workers, _queue_maxsize, _last_snapshot or {})


def try_acquire_slot() -> bool:
    if _queue_maxsize <= 0:
        return True
    with _pending_lock:
        global _pending_jobs
        if _pending_jobs >= _queue_maxsize:
            return False
        _pending_jobs += 1
        return True


def release_slot() -> int:
    with _pending_lock:
        global _pending_jobs
        if _pending_jobs > 0:
            _pending_jobs -= 1
        return _pending_jobs


def pending_jobs() -> int:
    with _pending_lock:
        return _pending_jobs


def submit_inference_job(payload: Dict[str, Any]) -> Future:
    if _pool is None:
        raise RuntimeError("Process pool not initialized")
    payload.setdefault("model_snapshot", _last_snapshot)
    try:
        return _pool.submit(_run_inference_job, payload)
    except BrokenProcessPool:
        # Recreate the pool and retry once
        _reset_pool()
        ensure_process_pool(_queue_workers, _queue_maxsize, _last_snapshot or {})
        if _pool is None:
            raise RuntimeError("Failed to recreate process pool")
        return _pool.submit(_run_inference_job, payload)


def _reset_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


# ---------------------------------------------------------------------------
# Job payload helpers
# ---------------------------------------------------------------------------

def build_job_payload(
    prepared_df: Optional[pd.DataFrame],
    inference_length: Optional[int],
    req_id: str,
    model_snapshot: Dict[str, Any],
    *,
    expected_base_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "prepared_df": prepared_df,
        "inference_length": inference_length,
        "req_id": req_id,
        "model_snapshot": model_snapshot,
        "timestamp": time.time(),
        "expected_base_columns": expected_base_columns or [],
    }


# ---------------------------------------------------------------------------
# Worker-side logic
# ---------------------------------------------------------------------------

_worker_inferencer = None
_worker_lock = threading.Lock()
_worker_snapshot: Dict[str, Any] | None = None
_worker_main_module = None
_worker_main_lock = threading.Lock()


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
        cleaned = raw.replace("not found in axis", "").strip(" ' \"")
        if cleaned:
            return cleaned
    return str(exc)


def _load_worker_main():
    global _worker_main_module
    if _worker_main_module is not None:
        return _worker_main_module
    with _worker_main_lock:
        if _worker_main_module is not None:
            return _worker_main_module
        try:
            module = importlib.import_module("inference_container.main")
        except ModuleNotFoundError:
            module_dir = Path(__file__).resolve().parent
            package_name = "inference_container"
            package = sys.modules.get(package_name)
            if package is None:
                package = types.ModuleType(package_name)
                package.__path__ = [str(module_dir)]  # type: ignore[attr-defined]
                sys.modules[package_name] = package
            spec = importlib.util.spec_from_file_location(
                f"{package_name}.main", module_dir / "main.py"
            )
            if spec is None or spec.loader is None:
                raise
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            setattr(package, "main", module)
        _worker_main_module = module
        return module


def _worker_initializer(snapshot: Dict[str, Any]) -> None:
    try:
        global _worker_inferencer, _worker_snapshot
        _worker_snapshot = snapshot or {}
        if os.getenv("USE_DUMMY_INFERENCER", "0") in {"1", "true", "TRUE"}:
            _worker_inferencer = _create_dummy_inferencer()
            return
        worker_main = _load_worker_main()

        _worker_inferencer = worker_main.inferencer
        _prepare_worker_model(worker_main, snapshot)
    except Exception as exc:  # pragma: no cover - failsafe visibility
        try:
            print({
                "service": "inference",
                "event": "worker_initializer_failed",
                "error": repr(exc),
            }, flush=True)
        except Exception:
            pass
        raise


def _prepare_worker_model(worker_main, snapshot: Optional[Dict[str, Any]]) -> None:
    if _worker_inferencer is None:
        return
    try:
        worker_main._load_promoted_pointer(_worker_inferencer)
    except Exception:
        pass
    if getattr(_worker_inferencer, "current_model", None) is None:
        try:
            worker_main._attempt_load_promoted(_worker_inferencer)
        except Exception:
            pass
    if getattr(_worker_inferencer, "current_model", None) is None:
        _fallback_load_latest_model_worker(_worker_inferencer)


def _ensure_worker_inferencer(snapshot: Optional[Dict[str, Any]]) -> Any:
    global _worker_inferencer, _worker_snapshot
    if os.getenv("USE_DUMMY_INFERENCER", "0") in {"1", "true", "TRUE"}:
        return _worker_inferencer
    worker_main = _load_worker_main()

    with _worker_lock:
        if _worker_inferencer is None:
            _worker_inferencer = worker_main.inferencer
        if snapshot and snapshot != _worker_snapshot:
            _worker_snapshot = snapshot
            _prepare_worker_model(worker_main, snapshot)
        if getattr(_worker_inferencer, "current_model", None) is None:
            _prepare_worker_model(worker_main, snapshot)
        if getattr(_worker_inferencer, "current_model", None) is None:
            raise InferenceHTTPError(status_code=503, detail="Model not loaded yet", worker_id=os.getpid())
        return _worker_inferencer


def _fallback_load_latest_model_worker(service) -> None:
    try:
        from inference_container.main import _enrich_loaded_model  # type: ignore
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow import pyfunc

        client = MlflowClient()
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        if hasattr(client, "search_experiments"):
            experiments = client.search_experiments()
        elif hasattr(client, "list_experiments"):
            experiments = client.list_experiments()
        else:
            experiments = []
        exp_ids = [e.experiment_id for e in experiments] or ["0"]
        promoted = client.search_runs(
            experiment_ids=exp_ids,
            filter_string="tags.promoted = 'true' and attributes.status = 'FINISHED'",
            order_by=["attributes.end_time DESC"],
            max_results=1,
        )
        chosen = promoted[0] if promoted else None
        if chosen is None:
            any_runs = client.search_runs(
                experiment_ids=exp_ids,
                filter_string="attributes.status = 'FINISHED'",
                order_by=["attributes.end_time DESC"],
                max_results=1,
            )
            if any_runs:
                chosen = any_runs[0]
        if chosen is None:
            return
        run_id = chosen.info.run_id
        params = chosen.data.params or {}
        model_type = params.get("model_type") or params.get("MODEL_TYPE") or "model"
        model_uri_candidates = [f"runs:/{run_id}/{model_type}"]
        if model_type != "model":
            model_uri_candidates.append(f"runs:/{run_id}/model")
        for cand in model_uri_candidates:
            try:
                mdl = pyfunc.load_model(cand)
                service.current_model = mdl
                service.current_run_id = run_id
                service.current_run_name = model_type
                service.model_type = model_type
                service.current_config_hash = params.get("config_hash") or params.get("CONFIG_HASH")
                _enrich_loaded_model(service, run_id, model_type)
                break
            except Exception:
                continue
    except Exception:
        pass


def _create_dummy_inferencer():
    class _Dummy:
        def __init__(self):
            self.current_model = True
            self.current_run_id = os.getenv("DUMMY_RUN_ID", "dummy-run")
            self.model_type = os.getenv("DUMMY_MODEL_NAME", "dummy-model")
            self.current_config_hash = os.getenv("DUMMY_CONFIG_HASH", "hash")
            self.last_prediction_response = None

        def perform_inference(self, df, inference_length=1):
            import numpy as np
            rows = max(1, inference_length)
            idx = pd.date_range(start="2020-01-01", periods=rows, freq="min")
            values = np.ones(rows)
            return pd.DataFrame({"value": values}, index=idx)

    return _Dummy()


def _run_inference_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    if os.getenv("USE_DUMMY_INFERENCER", "0") in {"1", "true", "TRUE"}:
        return _run_dummy_inference(payload)

    req_id = payload.get("req_id")
    inference_length = payload.get("inference_length")
    snapshot = payload.get("model_snapshot")
    worker_started_at = time.time()
    service = _ensure_worker_inferencer(snapshot)

    prepared_df = payload.get("prepared_df")
    if prepared_df is None:
        if getattr(service, "df", None) is None:
            raise InferenceHTTPError(status_code=400, detail="No cached dataframe available and no data provided.", worker_id=os.getpid())
        df = service.df
    else:
        if not isinstance(prepared_df, pd.DataFrame):
            raise InferenceHTTPError(status_code=400, detail="Invalid dataframe payload for inference", worker_id=os.getpid())
        df = prepared_df

    expected_base = payload.get("expected_base_columns") or []
    if expected_base:
        missing_expected = [col for col in expected_base if col not in df.columns]
        if missing_expected:
            detail = ", ".join(sorted(missing_expected))
            raise InferenceHTTPError(status_code=400, detail=f"Missing required columns for inference: {detail}", worker_id=os.getpid())

    if not isinstance(df.index, pd.DatetimeIndex):
        raise InferenceHTTPError(status_code=400, detail="Inference dataframe index must be datetime", worker_id=os.getpid())

    eff_len = inference_length if inference_length is not None else 1
    _worker_log(
        "queue_job_start",
        req_id=req_id,
        inference_length=int(eff_len),
        model_type=getattr(service, "model_type", None),
    )
    _worker_log("predict_inference_start", req_id=req_id, inference_length=int(eff_len))

    try:
        maybe_delay = getattr(service, "simulate_delay_if_enabled", None)
        if callable(maybe_delay):
            asyncio.run(maybe_delay())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(maybe_delay())
        finally:
            loop.close()
    except Exception:
        pass

    result_df: Optional[pd.DataFrame] = None
    try:
        result_df = service.perform_inference(df, inference_length=eff_len)
    except InferenceHTTPError as exc:
        _worker_log("predict_inference_failed", req_id=req_id, error=exc.detail or "InferenceHTTPError")
        setattr(exc, "worker_id", os.getpid())
        raise
    except KeyError as exc:
        missing_cols = _format_missing_columns_error(exc)
        detail = f"Missing required columns for inference: {missing_cols}" if missing_cols else "Missing required columns for inference"
        err = InferenceHTTPError(status_code=400, detail=detail, worker_id=os.getpid())
        _worker_log("predict_inference_failed", req_id=req_id, error=detail, status_code=400)
        raise err
    except Exception as exc:
        err = InferenceHTTPError(status_code=500, detail=f"Inference execution failed: {exc}", worker_id=os.getpid())
        _worker_log("predict_inference_failed", req_id=req_id, error=str(exc), status_code=500)
        raise err from exc
    if result_df is None:
        err = InferenceHTTPError(status_code=500, detail="Inference skipped (see server logs)", worker_id=os.getpid())
        _worker_log("predict_inference_failed", req_id=req_id, error="inference_skipped", status_code=500)
        raise err

    identifier = os.getenv("IDENTIFIER") or "default"
    cols = ["value"] if "value" in result_df.columns else result_df.columns.tolist()
    pred_list = []
    for ts, row in result_df[cols].iterrows():
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
                elif pd.isna(val):
                    entry[c] = None
                else:
                    entry[c] = float(val)
            except Exception:
                entry[c] = None
        pred_list.append(entry)

    response = {
        "status": "SUCCESS",
        "identifier": identifier,
        "run_id": getattr(service, "current_run_id", None),
        "predictions": pred_list,
    }
    try:
        service.last_prediction_response = response
    except Exception:
        pass
    _worker_log("predict_inference_end", req_id=req_id, rows=len(pred_list))
    worker_finished_at = time.time()
    worker_duration_ms = int((worker_finished_at - worker_started_at) * 1000)
    _worker_log(
        "queue_job_done",
        req_id=req_id,
        duration_ms=worker_duration_ms,
        predictions=len(pred_list),
    )

    meta = {
        "worker_id": os.getpid(),
        "started_at": worker_started_at,
        "finished_at": worker_finished_at,
        "duration_ms": worker_duration_ms,
        "predictions": len(pred_list),
        "inference_length": int(eff_len),
    }
    return {"response": response, "meta": meta}


def _run_dummy_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    worker_started_at = time.time()
    req_id = payload.get("req_id")
    rows = max(1, payload.get("inference_length") or 1)
    _worker_log("queue_job_start", req_id=req_id, inference_length=rows, model_type="dummy")
    _worker_log("predict_inference_start", req_id=req_id, inference_length=rows)
    time.sleep(float(os.getenv("DUMMY_SLEEP_SECS", "0")))
    idx = pd.date_range(start="2020-01-01", periods=rows, freq="min")
    pred_list = [{"ts": ts.isoformat(), "value": float(i)} for i, ts in enumerate(idx)]
    response = {
        "status": "SUCCESS",
        "identifier": os.getenv("IDENTIFIER") or "default",
        "run_id": os.getenv("DUMMY_RUN_ID", "dummy-run"),
        "predictions": pred_list,
        "model_name": os.getenv("DUMMY_MODEL_NAME", "dummy-model"),
        "req_id": req_id,
    }
    worker_finished_at = time.time()
    worker_duration_ms = int((worker_finished_at - worker_started_at) * 1000)
    _worker_log("predict_inference_end", req_id=req_id, rows=len(pred_list))
    _worker_log("queue_job_done", req_id=req_id, duration_ms=worker_duration_ms, predictions=len(pred_list))
    meta = {
        "worker_id": os.getpid(),
        "started_at": worker_started_at,
        "finished_at": worker_finished_at,
        "duration_ms": worker_duration_ms,
        "predictions": len(pred_list),
        "inference_length": rows,
    }
    return {"response": response, "meta": meta}
