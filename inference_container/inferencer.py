# Phase-1 Optimization: Thread tuning to prevent CPU oversubscription
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def _log_level() -> str:
    """Get configured log level (info, debug, error). Default: info."""
    return os.getenv("INFERENCE_LOG_LEVEL", "info").lower()

def _should_log_debug() -> bool:
    """Check if debug-level logs should be emitted."""
    return _log_level() == "debug"

from client_utils import post_file
from data_utils import window_data, check_uniform, time_to_feature, subset_scaler, _fix_zero_scale
from kafka_utils import produce_message, publish_error
from trace_utils import trace_df_operation, trace_dataframe, trace_operation, trace_error, TRACE_ENABLED
import numpy as np
import pandas as pd
import mlflow
from mlflow.artifacts import download_artifacts  # type: ignore
import pickle
import tempfile
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, Union, Dict, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

# Phase-1 Optimization: Apply torch thread limits
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print({"service": "inference", "event": "thread_tuning_applied", "omp": 1, "mkl": 1, "torch_threads": 1})
except ImportError:
    pass

# Phase-2 Optimization: Shared async executor for logging (removes MinIO/Kafka from hot path)
_ASYNC_LOG_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="async-log")

# Phase-2 Optimization: Enable/disable features via env vars
_ENABLE_QUANTIZATION = os.environ.get("INFERENCE_ENABLE_QUANTIZATION", "0").lower() in {"1", "true", "yes"}
_ENABLE_ONNX = os.environ.get("INFERENCE_ENABLE_ONNX", "1").lower() in {"1", "true", "yes"}

if _should_log_debug():
    print({"service": "inference", "event": "phase2_optimization_config", "quantization_enabled": _ENABLE_QUANTIZATION, "onnx_enabled": _ENABLE_ONNX})

# Phase-3 Optimization: Zero-queue opportunistic batching (no buffering, no waiting)
_ENABLE_OPPORTUNISTIC_BATCHING = os.environ.get("ENABLE_OPPORTUNISTIC_BATCHING", "1").lower() in {"1", "true", "yes"}
_BATCH_COORDINATOR_LOCK = threading.Lock()
_PENDING_BATCH: List[Tuple[pd.DataFrame, int, threading.Event, dict]] = []  # [(df, inf_len, event, result_container)]
_BATCH_PROCESSING = False

if _should_log_debug():
    print({"service": "inference", "event": "phase3_optimization_config", "opportunistic_batching_enabled": _ENABLE_OPPORTUNISTIC_BATCHING})

# Constants - These should all be defined by the service later
TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]
SAMPLE_IDX = int(os.environ.get("SAMPLE_IDX", 0))
INFERENCE_LENGTH = int(os.environ.get("INFERENCE_LENGTH", 1))

INFER_VERSION = "infer_v20251002_04"

logger = logging.getLogger("inference.inferencer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def apply_jit_compilation(model, model_class: str, run_id: str, model_type: str) -> None:
    """Phase-1 Optimization: Apply PyTorch JIT compilation to loaded model if applicable.
    
    This is a helper function to apply JIT compilation after mlflow.pyfunc.load_model() calls.
    Safe to call multiple times - will skip if not PyTorch or already compiled.
    Auto-detects PyTorch models if model_class not yet set.
    """
    if _should_log_debug():
        print({"service": "inference", "event": "jit_check_start", "run_id": run_id, "model_type": model_type, "has_impl": hasattr(model, "_model_impl")})
    if not hasattr(model, "_model_impl"):
        return
    
    try:
        import torch
        
        # Find the PyTorch model - try multiple paths
        inner_model = None
        
        # Path 1: model._model_impl.pytorch_model (MLflow PyTorch flavor)
        if hasattr(model._model_impl, "pytorch_model"):
            inner_model = model._model_impl.pytorch_model
        # Path 2: model._model_impl.python_model.model (custom wrapper)
        elif hasattr(model._model_impl, "python_model") and hasattr(model._model_impl.python_model, "model"):
            inner_model = model._model_impl.python_model.model
        # Path 3: Direct access
        elif hasattr(model._model_impl, "model"):
            inner_model = model._model_impl.model
        
        if inner_model is None:
            print({"service": "inference", "event": "jit_model_not_found", "run_id": run_id, "model_type": model_type})
            return
        
        # Auto-detect if PyTorch model (check if has nn.Module methods)
        is_pytorch = hasattr(inner_model, "eval") and hasattr(inner_model, "parameters")
        if _should_log_debug():
            print({"service": "inference", "event": "jit_pytorch_detection", "run_id": run_id, "is_pytorch": is_pytorch, "has_eval": hasattr(inner_model, "eval"), "has_parameters": hasattr(inner_model, "parameters")})
        if not is_pytorch:
            return
        
        # Check if already JIT-compiled
        if isinstance(inner_model, torch.jit.ScriptModule):
            if _should_log_debug():
                print({"service": "inference", "event": "jit_already_compiled", "run_id": run_id, "model_type": model_type})
            return
        
        # Apply JIT compilation
        inner_model.eval()
        try:
            jit_model = torch.jit.script(inner_model)
            # Update the reference based on which path we found
            if hasattr(model._model_impl, "pytorch_model"):
                model._model_impl.pytorch_model = jit_model
            elif hasattr(model._model_impl, "python_model") and hasattr(model._model_impl.python_model, "model"):
                model._model_impl.python_model.model = jit_model
            elif hasattr(model._model_impl, "model"):
                model._model_impl.model = jit_model
            # Keep JIT success at info level
            print({"service": "inference", "event": "jit_compiled", "run_id": run_id, "model_type": model_type})
        except Exception as jit_err:
            print({"service": "inference", "event": "jit_compile_failed", "run_id": run_id, "model_type": model_type, "error": str(jit_err)})
    except Exception as jit_outer_err:
        print({"service": "inference", "event": "jit_setup_failed", "run_id": run_id, "error": str(jit_outer_err)})


def _read_simulated_delay() -> float:
    raw = os.getenv("SIMULATE_DELAY_SECS", "0")
    if raw is None:
        return 0.0
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Invalid SIMULATE_DELAY_SECS value '%s'; defaulting to 0", raw)
        return 0.0
    return max(0.0, value)


SIMULATE_DELAY_SECS = _read_simulated_delay()

class Inferencer:
    def __init__(self, gateway_url: str, producer, dlq_topic: str, output_topic: str):
        self.gateway_url = gateway_url
        self.producer = producer
        self.dlq_topic = dlq_topic
        self.output_topic = output_topic
        self.df = None
        # Lock to protect concurrent access / mutation of self.df
        self._df_lock = threading.RLock()
        self.input_seq_len = 0
        self.output_seq_len = 0
        self.current_model = None
        self.current_scaler: Union[MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, None] = None
        self.current_experiment_name = "Default"
        self.current_run_name = ""
        self.model_type = ""
        self.model_class = ""  # "pytorch", "prophet", "statsforecast"
        # Track emitted (run_id, prediction_hash) to prevent duplicate JSONL rows
        self._emitted_prediction_keys: set[tuple[str, str]] = set()
        self._emitted_prediction_lock = threading.Lock()
        # Lock to protect reads/writes to last_prediction_response
        self._last_prediction_lock = threading.Lock()
        # Track active inference jobs for visibility/metrics without blocking concurrency
        self._active_jobs = 0
        self._active_jobs_lock = threading.Lock()
        # Track which run_ids we've already attempted scaler resolution for (prevents spammy logs)
        self._scaler_checked_run_ids = set()
        self.simulate_delay_secs = SIMULATE_DELAY_SECS
        self._last_inference_timings = None
        # Phase-2 Optimization: Track quantization and ONNX status
        self.quantization_applied = False
        self.onnx_model = None
        self.onnx_session = None
        self.onnx_cache_path = None  # Phase-4: Persistent ONNX cache
        self.model_backend = "pytorch"  # Phase-4: Track active backend (pytorch/onnx)
        logger.info("SIMULATE_DELAY_SECS=%s", self.simulate_delay_secs)

    # ----------------- Thread-safe helpers for shared state -----------------
    def get_df_copy(self) -> Optional[pd.DataFrame]:
        """Return a deep copy of the current service dataframe or None.

        Uses an RLock to ensure a consistent snapshot is returned.
        """
        with self._df_lock:
            if self.df is None:
                return None
            try:
                return self.df.copy(deep=True)
            except Exception:
                # Best-effort fallback to shallow copy if deep fails
                return self.df.copy()

    def set_df(self, df: Optional[pd.DataFrame]) -> None:
        """Atomically replace the service dataframe."""
        with self._df_lock:
            self.df = df

    def get_last_prediction_copy(self) -> Optional[dict]:
        with self._last_prediction_lock:
            val = getattr(self, "last_prediction_response", None)
            if val is None:
                return None
            try:
                return val.copy()
            except Exception:
                return dict(val)

    def set_last_prediction(self, payload: dict) -> None:
        with self._last_prediction_lock:
            try:
                self.last_prediction_response = payload.copy()
            except Exception:
                self.last_prediction_response = dict(payload)

    async def simulate_delay_if_enabled(self) -> None:
        delay = _read_simulated_delay()
        self.simulate_delay_secs = delay
        if delay <= 0:
            return
        logger.info("Simulating inference delay: %ss", delay)
        await asyncio.sleep(delay)

    # Phase-3 Optimization: Zero-queue opportunistic batching
    def coordinate_batch_inference(self, df: pd.DataFrame, inference_length: int) -> Optional[pd.DataFrame]:
        """Coordinate opportunistic batching with zero wait time.
        
        Only batches requests that arrive in the same event-loop tick.
        If no other requests are pending, executes immediately without batching.
        
        Returns:
            Result DataFrame or None if batching is in progress (will be filled by batch leader)
        """
        global _BATCH_COORDINATOR_LOCK, _PENDING_BATCH, _BATCH_PROCESSING
        
        if not _ENABLE_OPPORTUNISTIC_BATCHING:
            # Batching disabled, execute normally
            return self.perform_inference(df, inference_length=inference_length)
        
        result_container = {}
        event = threading.Event()
        
        with _BATCH_COORDINATOR_LOCK:
            # Add this request to pending batch
            _PENDING_BATCH.append((df, inference_length, event, result_container))
            
            if _BATCH_PROCESSING:
                # Another request is already processing batch, wait for it
                batch_id = f"batch-wait-{threading.current_thread().ident}"
                print({"service": "inference", "event": "opportunistic_batch_join", "batch_id": batch_id, "pending_count": len(_PENDING_BATCH)}, flush=True)
            else:
                # We're the first request, become batch leader
                _BATCH_PROCESSING = True
                batch_id = f"batch-lead-{threading.current_thread().ident}"
                
                # Immediately snapshot pending requests (zero wait time)
                batch_snapshot = _PENDING_BATCH[:]
                _PENDING_BATCH.clear()
                
                batch_size = len(batch_snapshot)
                print({"service": "inference", "event": "opportunistic_batch_execute", "batch_id": batch_id, "batch_size": batch_size}, flush=True)
                
                # Release lock before inference (don't block other requests)
                _BATCH_COORDINATOR_LOCK.release()
                
                try:
                    if batch_size == 1:
                        # Only one request, execute normally (no batching overhead)
                        single_df, single_len, single_event, single_result = batch_snapshot[0]
                        result = self.perform_inference(single_df, inference_length=single_len)
                        single_result['df'] = result
                        single_event.set()
                        print({"service": "inference", "event": "opportunistic_batch_single", "batch_id": batch_id}, flush=True)
                        return result
                    
                    # Multiple requests: batch them
                    # Concatenate DataFrames (preserving individual request boundaries)
                    batch_dfs = []
                    batch_lens = []
                    for req_df, req_len, _, _ in batch_snapshot:
                        batch_dfs.append(req_df)
                        batch_lens.append((len(req_df), req_len))
                    
                    # Stack DataFrames into single batch
                    combined_df = pd.concat(batch_dfs, axis=0, ignore_index=False)
                    max_inference_len = max(req_len for _, req_len in batch_lens)
                    
                    print({"service": "inference", "event": "opportunistic_batch_combined", 
                           "batch_id": batch_id, "total_rows": len(combined_df), 
                           "max_inference_len": max_inference_len}, flush=True)
                    
                    # Execute batched inference
                    batch_result = self.perform_inference(combined_df, inference_length=max_inference_len)
                    
                    # Split results back to individual requests
                    if batch_result is not None:
                        current_idx = 0
                        for i, ((req_rows, req_len), (_, _, req_event, req_result)) in enumerate(zip(batch_lens, batch_snapshot)):
                            # Extract rows belonging to this request
                            individual_result = batch_result.iloc[current_idx:current_idx + req_rows].copy()
                            req_result['df'] = individual_result
                            req_event.set()
                            current_idx += req_rows
                    else:
                        # Inference failed, signal all requests with None
                        for _, _, req_event, req_result in batch_snapshot:
                            req_result['df'] = None
                            req_event.set()
                    
                    print({"service": "inference", "event": "opportunistic_batch_complete", 
                           "batch_id": batch_id, "batch_size": batch_size}, flush=True)
                    
                    # Return our own result
                    return result_container.get('df')
                    
                finally:
                    # Re-acquire lock to reset processing flag
                    _BATCH_COORDINATOR_LOCK.acquire()
                    _BATCH_PROCESSING = False
        
        # We joined an existing batch, wait for results
        event.wait(timeout=30)  # 30s timeout to prevent deadlock
        result = result_container.get('df')
        
        if result is None:
            print({"service": "inference", "event": "opportunistic_batch_timeout", "batch_id": batch_id}, flush=True)
        
        return result

    def load_model(self, experiment_name: str, run_name: str, sort: str="Recent"):
        if _should_log_debug():
            print(f"[Inferencer:{INFER_VERSION}] Attempting to load model for experiment: {experiment_name}, run: {run_name}")

        try:
            if sort == "Recent":
                order = ["start_time desc"]
            elif sort == "Best":
                order = ["mse desc"] # not sure if this is correct
            else:
                raise TypeError("Invalid sort argument")
            
            runs_df: pd.DataFrame = mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string=f"tags.mlflow.runName = '{run_name}'", # Filter by run name
                order_by=order,
                max_results=1,
                output_format="pandas"
            ) # type: ignore (output_format="pandas" ensures we get a DataFrame)

            if runs_df.empty:
                raise Exception(f"No runs found for experiment '{experiment_name}' with run name '{run_name}'.")

            run_id = runs_df.loc[0, "run_id"]
            self.current_run_id = run_id

                    # Extract model parameters and store them in self.params
            run_row = runs_df.iloc[0]
            self.params = {}
            for col in run_row.index:
                if col.startswith("params."):
                    param_name = col.replace("params.", "")
                    self.params[param_name] = run_row[col]
            
            if _should_log_debug():
                print(f"Extracted parameters: {self.params}")
            
            # Detect model type from experiment name or parameters
            self.model_type, self.model_class = self._detect_model_type(runs_df.iloc[0])
            
            if _should_log_debug():
                print(f"Found run with ID: {run_id}, Model type: {self.model_type}, Model class: {self.model_class}")

            if self.model_class == "pytorch":
                self.input_seq_len = int(runs_df["params.input_sequence_length"][0])
                self.output_seq_len = int(runs_df["params.output_sequence_length"][0])

            base_uri = f"runs:/{run_id}"
            # Candidate artifact subpaths to try (ordered). run_name first (legacy), then 'model' (mlflow.autolog default)
            candidates = [run_name, "model"]
            # De-duplicate if run_name already 'model'
            seen = set()
            candidates = [c for c in candidates if not (c in seen or seen.add(c))]

            last_err: Optional[Exception] = None
            for subpath in candidates:
                candidate_uri = f"{base_uri}/{subpath}"
                try:
                    print(f"Attempting to load model from: {candidate_uri}")
                    reqs = mlflow.pyfunc.get_model_dependencies(candidate_uri)
                    print(f"Model dependencies (candidate='{subpath}'): {reqs}")
                    model = mlflow.pyfunc.load_model(candidate_uri)
                    self.current_model = model
                    
                    # Phase-1 Optimization: Apply PyTorch JIT compilation if applicable
                    if self.model_class == "pytorch" and hasattr(model, "_model_impl"):
                        try:
                            import torch
                            wrapped_model = getattr(model._model_impl, "python_model", None)
                            if wrapped_model is not None and hasattr(wrapped_model, "model"):
                                inner_model = wrapped_model.model
                                if hasattr(inner_model, "eval"):
                                    inner_model.eval()
                                    try:
                                        jit_model = torch.jit.script(inner_model)
                                        wrapped_model.model = jit_model
                                        print({"service": "inference", "event": "jit_compiled", "run_id": run_id, "model_type": self.model_type})
                                    except Exception as jit_err:
                                        print({"service": "inference", "event": "jit_compile_failed", "run_id": run_id, "model_type": self.model_type, "error": str(jit_err)})
                        except Exception as jit_outer_err:
                            print({"service": "inference", "event": "jit_setup_failed", "run_id": run_id, "error": str(jit_outer_err)})
                    
                    # Phase-2 Optimization: Apply dynamic quantization (INT8)
                    if self.model_class == "pytorch" and _ENABLE_QUANTIZATION and hasattr(model, "_model_impl"):
                        self._apply_quantization(model, run_id)
                    
                    # Phase-2 Optimization: Convert to ONNX Runtime if enabled
                    if self.model_class == "pytorch" and _ENABLE_ONNX and hasattr(model, "_model_impl"):
                        self._convert_to_onnx(model, run_id)
                    
                    # Phase-4: Log backend selection
                    backend_info = {
                        "service": "inference",
                        "event": "model_backend_selected",
                        "run_id": run_id,
                        "model_type": self.model_type,
                        "model_class": self.model_class,
                        "backend": self.model_backend,
                        "quantization_applied": self.quantization_applied,
                        "onnx_enabled": self.onnx_session is not None
                    }
                    # Keep backend info at info level
                    print(backend_info)
                    
                    self.current_experiment_name = experiment_name
                    self.current_run_name = run_name
                    if _should_log_debug():
                        print(f"Model loaded successfully from subpath '{subpath}'.")
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"Model load attempt failed for subpath '{subpath}': {e}")
                    last_err = e
                    continue

            if self.current_model is None:
                # Exhausted candidates - list artifacts at root for debug
                try:
                    from mlflow.tracking import MlflowClient  # type: ignore
                    client = MlflowClient()
                    arts = client.list_artifacts(run_id, path="")
                    print({"service": "inference", "event": "artifact_root_list_on_fail", "run_id": run_id, "items": [a.path for a in arts]})
                except Exception as le:  # noqa: BLE001
                    print({"service": "inference", "event": "artifact_list_fail", "run_id": run_id, "error": str(le)})
                raise last_err or Exception("Unknown model load failure (no candidates tried)")

            # Attempt to load scaler artifact (optional). Flexible discovery:
            # 1. Any *.pkl file directly under 'scaler/' artifact directory (preferred if name contains 'scaler')
            # 2. Any *.pkl at artifact root with 'scaler' in its name
            # Avoid repeated warnings by caching run_ids we've inspected.
            if run_id in self._scaler_checked_run_ids and self.current_scaler is None:
                if _should_log_debug():
                    print("[Info] Skipping scaler search (previously not found for this run).")
            elif self.current_scaler is not None and run_id in self._scaler_checked_run_ids:
                # Already loaded earlier; nothing to do
                pass
            else:
                scaler_loaded = False
                try:
                    from mlflow.tracking import MlflowClient  # type: ignore
                    client = MlflowClient()
                    # List artifacts under 'scaler' directory first
                    try:
                        scaler_dir_items = client.list_artifacts(run_id, path="scaler")
                    except Exception:
                        scaler_dir_items = []
                    pkl_candidates = [a.path for a in scaler_dir_items if not getattr(a, 'is_dir', False) and a.path.lower().endswith('.pkl')]
                    # If nothing found in scaler/, look at root for any scaler-related pkl
                    if not pkl_candidates:
                        try:
                            root_items = client.list_artifacts(run_id, path="")
                        except Exception:
                            root_items = []
                        root_pkls = [a.path for a in root_items if not getattr(a, 'is_dir', False) and a.path.lower().endswith('.pkl')]
                        # Prefer names containing 'scaler'
                        root_pkls_sorted = sorted(root_pkls, key=lambda n: (0 if 'scaler' in n.lower() else 1, len(n)))
                        # Keep only scaler-related first if exists
                        if root_pkls_sorted:
                            pkl_candidates = root_pkls_sorted
                    # Rank candidates: contain 'scaler' first, then shorter name
                    pkl_candidates = sorted(pkl_candidates, key=lambda n: (0 if 'scaler' in n.lower() else 1, len(n)))
                    if pkl_candidates:
                        chosen_rel_path = pkl_candidates[0]
                        scaler_artifact_uri = f"{base_uri}/{chosen_rel_path}"
                        try:
                            scaler_path = download_artifacts(artifact_uri=scaler_artifact_uri, dst_path=tempfile.gettempdir())
                            with open(scaler_path, "rb") as f:
                                self.current_scaler = pickle.load(f)
                            # Apply zero-scale fix to prevent division-by-zero during inverse_transform
                            if self.current_scaler is not None:
                                scaler_type_name = self.current_scaler.__class__.__name__
                                self.current_scaler = _fix_zero_scale(self.current_scaler, scaler_type_name=scaler_type_name)
                            if self.current_scaler is not None:
                                scaler_loaded = True
                                print({
                                    "service": "inference",
                                    "event": "scaler_loaded",
                                    "run_id": run_id,
                                    "artifact_path": chosen_rel_path
                                })
                        except Exception as le:  # noqa: BLE001
                            print({
                                "service": "inference",
                                "event": "scaler_load_failed",
                                "run_id": run_id,
                                "artifact_path": chosen_rel_path,
                                "error": str(le)
                            })
                    if not scaler_loaded:
                        print({
                            "service": "inference",
                            "event": "scaler_not_found",
                            "run_id": run_id,
                            "note": "No scaler .pkl located under scaler/ or root; proceeding without scaler"
                        })
                except Exception as se:  # noqa: BLE001
                    print({
                        "service": "inference",
                        "event": "scaler_search_error",
                        "run_id": run_id,
                        "error": str(se)
                    })
                finally:
                    self._scaler_checked_run_ids.add(run_id)

        except Exception as e:
            print(f"Error loading model: {e}")
            publish_error(
                self.producer,
                self.dlq_topic,
                "Model Load",
                "Failure",
                str(e),
                {"experiment": experiment_name, "run_name": run_name}
            )

    def _apply_quantization(self, model, run_id: str) -> None:
        """Phase-2: Apply dynamic INT8 quantization with safe fallback."""
        try:
            import torch
            wrapped_model = getattr(model._model_impl, "python_model", None)
            if wrapped_model is None or not hasattr(wrapped_model, "model"):
                print({"service": "inference", "event": "quantization_skip_no_model", "run_id": run_id})
                return
            
            inner_model = wrapped_model.model
            if not hasattr(inner_model, "eval"):
                print({"service": "inference", "event": "quantization_skip_not_pytorch", "run_id": run_id})
                return
            
            inner_model.eval()
            # Apply dynamic quantization to Linear and LSTM/GRU layers
            quantized_model = torch.quantization.quantize_dynamic(
                inner_model,
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8
            )
            wrapped_model.model = quantized_model
            self.quantization_applied = True
            print({"service": "inference", "event": "quantization_applied", "run_id": run_id, "model_type": self.model_type, "dtype": "qint8"})
        except Exception as quant_err:
            self.quantization_applied = False
            print({"service": "inference", "event": "quantization_failed_fallback", "run_id": run_id, "error": str(quant_err)})
    
    def _convert_to_onnx(self, model, run_id: str) -> None:
        """Phase-4: Convert PyTorch model to ONNX Runtime with persistent cache."""
        try:
            import torch
            import tempfile
            import hashlib
            
            # Try importing onnxruntime and onnx
            try:
                import onnxruntime as ort
                import onnx
            except ImportError:
                print({"service": "inference", "event": "onnx_skip_not_installed", "run_id": run_id})
                return
            
            # Extract PyTorch model - try all paths like JIT compilation does
            inner_model = None
            
            # Path 1: model._model_impl.pytorch_model (MLflow PyTorch flavor)
            if hasattr(model._model_impl, "pytorch_model"):
                inner_model = model._model_impl.pytorch_model
            # Path 2: model._model_impl.python_model.model (custom wrapper)
            elif hasattr(model._model_impl, "python_model") and hasattr(model._model_impl.python_model, "model"):
                inner_model = model._model_impl.python_model.model
            # Path 3: Direct access
            elif hasattr(model._model_impl, "model"):
                inner_model = model._model_impl.model
            
            if inner_model is None or not hasattr(inner_model, "eval"):
                print({"service": "inference", "event": "onnx_skip_no_pytorch_model", "run_id": run_id})
                return
            
            inner_model.eval()
            
            # Create cache path based on run_id and model structure
            cache_dir = os.path.join(tempfile.gettempdir(), "onnx_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Generate deterministic cache key from model structure
            model_str = str(inner_model)
            cache_key = hashlib.md5(f"{run_id}_{model_str}".encode()).hexdigest()[:16]
            onnx_cache_path = os.path.join(cache_dir, f"{cache_key}.onnx")
            
            # Check if ONNX model already exists and is valid
            if os.path.exists(onnx_cache_path):
                try:
                    # Validate cached ONNX model
                    onnx_model = onnx.load(onnx_cache_path)
                    onnx.checker.check_model(onnx_model)
                    
                    # Load into ONNX Runtime
                    sess_options = ort.SessionOptions()
                    sess_options.intra_op_num_threads = 1
                    sess_options.inter_op_num_threads = 1
                    self.onnx_session = ort.InferenceSession(onnx_cache_path, sess_options)
                    self.onnx_cache_path = onnx_cache_path
                    self.model_backend = "onnx"
                    
                    print({"service": "inference", "event": "onnx_loaded_from_cache", "run_id": run_id, "cache_path": onnx_cache_path})
                    return
                except Exception as cache_err:
                    print({"service": "inference", "event": "onnx_cache_invalid_regenerating", "run_id": run_id, "error": str(cache_err)[:200]})
                    try:
                        os.unlink(onnx_cache_path)
                    except Exception:
                        pass
            
            # Create dummy input for export (batch=1, seq_len from params, features from model)
            input_seq_len = self.input_seq_len or 10
            # Infer feature count from model's first layer if possible
            feature_dim = 1
            try:
                if hasattr(inner_model, "lstm") and hasattr(inner_model.lstm, "input_size"):
                    feature_dim = inner_model.lstm.input_size
                elif hasattr(inner_model, "gru") and hasattr(inner_model.gru, "input_size"):
                    feature_dim = inner_model.gru.input_size
                elif hasattr(inner_model, "rnn") and hasattr(inner_model.rnn, "input_size"):
                    feature_dim = inner_model.rnn.input_size
            except Exception:
                pass
            
            dummy_input = torch.randn(1, input_seq_len, feature_dim)
            
            # Export to ONNX with persistent cache
            torch.onnx.export(
                inner_model,
                dummy_input,
                onnx_cache_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch", 1: "seq_len"}, "output": {0: "batch", 1: "seq_len"}}
            )
            
            # Validate exported ONNX model
            onnx_model = onnx.load(onnx_cache_path)
            onnx.checker.check_model(onnx_model)
            
            # Load into ONNX Runtime
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            self.onnx_session = ort.InferenceSession(onnx_cache_path, sess_options)
            self.onnx_cache_path = onnx_cache_path
            self.model_backend = "onnx"
            
            print({"service": "inference", "event": "onnx_export_success", "run_id": run_id, "model_type": self.model_type, "input_shape": [1, input_seq_len, feature_dim], "cache_path": onnx_cache_path})
        except Exception as onnx_err:
            self.onnx_session = None
            self.model_backend = "pytorch"
            print({"service": "inference", "event": "onnx_export_failed_fallback", "run_id": run_id, "error": str(onnx_err)[:500]})
    
    def _detect_model_type(self, run_row: pd.Series) -> Tuple[str, str]:
        """Detect [model_type, model_class] from MLflow run parameters or tags."""

        # Check for explicit model type parameter
        if "params.model_type" in run_row and pd.notna(run_row["params.model_type"]):
            model_type = run_row["params.model_type"].upper()
            if model_type in ["LSTM", "GRU", "TETS", "TCN"]:
                return model_type, "pytorch"
            elif model_type in ["AUTOARIMA", "AUTOETS", "AUTOTHETA", "AUTOMFLES", "AUTOTBATS"]:
                return model_type, "statsforecast"
            elif model_type == "PROPHET":
                return "PROPHET", "prophet"

        # Check experiment name
        exp_name = self.current_experiment_name.lower()
        if "prophet" in exp_name:
            return "PROPHET", "prophet"

        for sf_model in ["autoarima", "autoets", "autotheta", "automfles", "autotbats"]:
            if sf_model in exp_name:
                return sf_model.upper(), "statsforecast"

        for pt_model in ["lstm", "gru", "tets", "tcn"]:
            if pt_model in exp_name:
                return pt_model.upper(), "pytorch"

        # Check params to infer framework
        if any(param.startswith("params.seasonality") 
            for param in run_row.index if pd.notna(run_row.get(param))):
            return "PROPHET", "prophet"

        if any(param.startswith("params.season_length") 
            for param in run_row.index if pd.notna(run_row.get(param))):
            return "", "statsforecast"  # fallback default for statsforecast

        # Default fallback
        return "", "pytorch"

    def _mark_job_started(self) -> None:
        with self._active_jobs_lock:
            self._active_jobs += 1

    def _mark_job_finished(self) -> None:
        with self._active_jobs_lock:
            self._active_jobs = max(0, self._active_jobs - 1)

    @property
    def active_inference_jobs(self) -> int:
        with self._active_jobs_lock:
            return self._active_jobs

    @trace_df_operation
    def perform_inference(self, df_eval: Optional[pd.DataFrame] = None, inference_length: Optional[int] = None):
        """Execute inference.

        Parameters
        ----------
        df_eval : Optional[pd.DataFrame]
            Optional override dataframe. If None uses self.df.
        inference_length : Optional[int]
            Override number of forecast steps (defaults to env INFERENCE_LENGTH).

        Returns
        -------
        Optional[pd.DataFrame]
            Predictions dataframe (inverse scaled when possible) or None if skipped.
        """
        trace_operation("perform_inference_start", df_eval_provided=df_eval is not None, has_model=self.current_model is not None)
        
        if df_eval is None:
            if self.df is None:
                trace_error("perform_inference", ValueError("No data"), message="No data provided for inference and service dataframe is empty")
                if _should_log_debug():
                    print("No data provided for inference and service dataframe is empty.")
                return None
            # CRITICAL: Deep copy shared DataFrame to prevent concurrent modification
            df_eval = self.df.copy(deep=True)
            trace_dataframe("after_service_df_copy", df_eval, {"source": "self.df"}, "perform_inference")
        else:
            trace_dataframe("perform_inference_entry", df_eval, {"source": "request_override"}, "perform_inference")
            if os.getenv("INFER_VERBOSE_DATA", "0") in {"1","true","TRUE"}:
                try:
                    # Disable verbose data dumps
                    pass
                except Exception:
                    pass
        if self.current_model is None:
            trace_operation("no_model_loaded", action="defer_inference")
            if _should_log_debug():
                print("[INFO] Model not loaded yet. Deferring inference (no DLQ).")
            return None
        local_inference_length = int(inference_length) if inference_length is not None else INFERENCE_LENGTH
        self._mark_job_started()
        timings: Dict[str, float] = {}
        overall_start = time.perf_counter()

        def _finalize_timings() -> None:
            if "overall_ms" not in timings:
                timings["overall_ms"] = (time.perf_counter() - overall_start) * 1000.0
            # Phase-1: Filter out non-numeric metadata (e.g., feature_engineering_method)
            self._last_inference_timings = {k: float(v) for k, v in timings.items() if isinstance(v, (int, float))}

        print({"service": "inference", "event": "predict_inference_start", "inference_length": int(local_inference_length)})
        trace_operation("inference_params", input_seq_len=self.input_seq_len, output_seq_len=self.output_seq_len, inference_length=local_inference_length)
        try:
            total_rows = len(df_eval.index)
            min_needed = self.input_seq_len + self.output_seq_len
            if self.input_seq_len > 0 and total_rows < min_needed:
                print({
                    "service": "inference",
                    "event": "insufficient_rows",
                    "rows": int(total_rows),
                    "input_seq_len": int(self.input_seq_len),
                    "output_seq_len": int(self.output_seq_len),
                    "min_required": int(min_needed),
                    "action": "skip_inference"
                })
                _finalize_timings()
                return None
            required_index = SAMPLE_IDX + self.input_seq_len
            if total_rows == 0:
                if _should_log_debug():
                    print("[Inferencer] Empty dataframe passed to inference; aborting.")
                _finalize_timings()
                return None
            if required_index >= total_rows:
                adjusted_start_pos = total_rows - 1
                print({
                    "service": "inference",
                    "event": "adjust_start_index",
                    "reason": "index_out_of_bounds",
                    "requested_start_pos": int(required_index),
                    "adjusted_start_pos": int(adjusted_start_pos),
                    "input_seq_len": int(self.input_seq_len),
                    "sample_idx": int(SAMPLE_IDX),
                    "rows": int(total_rows)
                })
            else:
                adjusted_start_pos = required_index

            timings["precheck_ms"] = (time.perf_counter() - overall_start) * 1000.0

            # Phase-5: Start preprocessing timing
            timings["preprocessing_start"] = time.perf_counter()
            
            stage_start = time.perf_counter()
            # Diagnostic: Log first few timestamps to debug zero timedelta
            if len(df_eval.index) > 0:
                # Debug dumps disabled (enable via INFERENCE_LOG_LEVEL=debug)
                pass
            timedelta = check_uniform(df_eval)
            timings["check_uniform_ms"] = (time.perf_counter() - stage_start) * 1000.0

            start_timestamp = df_eval.index[adjusted_start_pos]
            stage_start = time.perf_counter()
            
            # Phase-1 Optimization: Fast NumPy-based feature engineering
            try:
                # Create timestamp index using numpy
                timestamps = pd.date_range(
                    start=start_timestamp,
                    periods=local_inference_length,
                    freq=timedelta
                )
                
                # Pre-allocate DataFrame with same columns
                df_predictions = pd.DataFrame(
                    index=timestamps,
                    columns=df_eval.columns
                )
                
                # Fast NumPy feature engineering (replaces time_to_feature pandas operations)
                ts_array = df_predictions.index.to_numpy(dtype="datetime64[m]")
                minutes_since_epoch = ts_array.astype("int64")
                
                # Minute of day features (0-1439)
                minutes = minutes_since_epoch % 1440
                df_predictions["min_of_day_sin"] = np.sin(minutes * 2 * np.pi / 1440)
                df_predictions["min_of_day_cos"] = np.cos(minutes * 2 * np.pi / 1440)
                
                # Day of week features (0-6)
                days_since_epoch = ts_array.astype("datetime64[D]").astype("int64")
                dow = (days_since_epoch + 4) % 7  # Adjust epoch offset to match pandas weekday
                df_predictions["day_of_week_sin"] = np.sin(dow * 2 * np.pi / 7)
                df_predictions["day_of_week_cos"] = np.cos(dow * 2 * np.pi / 7)
                
                # Day of year features (1-366)
                start_of_year = ts_array.astype("datetime64[Y]")
                days_in_year = (ts_array.astype("datetime64[D]") - start_of_year.astype("datetime64[D]")).astype("int64") + 1
                df_predictions["day_of_year_sin"] = np.sin(days_in_year * 2 * np.pi / 366)
                df_predictions["day_of_year_cos"] = np.cos(days_in_year * 2 * np.pi / 366)
                
                numpy_duration_ms = (time.perf_counter() - stage_start) * 1000.0
                timings["prepare_prediction_frame_ms"] = numpy_duration_ms
                timings["feature_engineering_ms"] = numpy_duration_ms  # Phase-5: Also track as feature engineering
                timings["feature_engineering_method"] = "numpy_optimized"
                
                print({
                    "service": "inference",
                    "event": "feature_engineering_optimized",
                    "duration_ms": round(numpy_duration_ms, 3),
                    "method": "numpy",
                    "rows": len(df_predictions)
                })
                
            except Exception as fe_err:
                # Fallback to original pandas implementation if numpy optimization fails
                print({"service": "inference", "event": "feature_engineering_fallback", "error": str(fe_err)})
                df_predictions = pd.DataFrame(
                    index=pd.date_range(
                        start=start_timestamp,
                        periods=local_inference_length,
                        freq=timedelta
                    ),
                    columns=df_eval.columns
                )
                
                # Phase-5: Time the fallback feature engineering
                fe_fallback_start = time.perf_counter()
                df_predictions = time_to_feature(df_predictions)
                timings["prepare_prediction_frame_ms"] = (time.perf_counter() - stage_start) * 1000.0
                timings["feature_engineering_ms"] = (time.perf_counter() - fe_fallback_start) * 1000.0
                timings["feature_engineering_method"] = "pandas_fallback"

            # Phase-5: Calculate total preprocessing time
            if "preprocessing_start" in timings:
                timings["preprocessing_total_ms"] = (time.perf_counter() - timings["preprocessing_start"]) * 1000.0
            
            branch_start = time.perf_counter()
            if self.model_class == "pytorch":
                df_transformed_predictions = self._perform_pytorch_inference(df_eval, df_predictions, local_inference_length, timings)
            elif self.model_class == "prophet":
                df_transformed_predictions = self._perform_prophet_inference(df_eval, df_predictions, local_inference_length, timings)
            elif self.model_class == "statsforecast":
                df_transformed_predictions = self._perform_statsforecast_inference(df_eval, df_predictions, local_inference_length, timings)
            else:
                raise ValueError(f"Unsupported model class: {self.model_class}")
            timings["model_branch_ms"] = (time.perf_counter() - branch_start) * 1000.0

            save_start = time.perf_counter()
            # Phase-2: Execute logging asynchronously (removes MinIO/Kafka from hot path)
            _ASYNC_LOG_EXECUTOR.submit(self._save_and_publish_predictions, df_transformed_predictions, df_eval, timings)
            timings["save_publish_ms"] = (time.perf_counter() - save_start) * 1000.0
            timings["async_logging_used"] = 1.0

            _finalize_timings()
            print({"service": "inference", "event": "predict_inference_end", "rows": int(df_transformed_predictions.shape[0])})
            
            # Emit comprehensive timing breakdown for bottleneck profiling
            try:
                timing_breakdown = {
                    "t_precheck_ms": round(timings.get("precheck_ms", 0.0), 3),
                    "t_check_uniform_ms": round(timings.get("check_uniform_ms", 0.0), 3),
                    "t_prepare_prediction_frame_ms": round(timings.get("prepare_prediction_frame_ms", 0.0), 3),
                    "t_window_data_ms": round(timings.get("window_data_ms", 0.0), 3),
                    "t_model_predict_ms": round(timings.get("model_predict_ms", 0.0), 3),
                    "t_pytorch_loop_ms": round(timings.get("pytorch_loop_ms", 0.0), 3),
                    "t_inverse_scale_ms": round(timings.get("inverse_scale_ms", 0.0), 3),
                    "t_save_publish_ms": round(timings.get("save_publish_ms", 0.0), 3),
                    "t_model_branch_ms": round(timings.get("model_branch_ms", 0.0), 3),
                    "t_total_ms": round(timings.get("overall_ms", 0.0), 3),
                    # Phase-5: Preprocessing timings
                    "t_preprocessing_total_ms": round(timings.get("preprocessing_total_ms", 0.0), 3),
                    "t_feature_engineering_ms": round(timings.get("feature_engineering_ms", 0.0), 3),
                    "t_window_preparation_ms": round(timings.get("window_preparation_ms", 0.0), 3),
                    "model_predict_calls": int(timings.get("model_predict_calls", 0)),
                    "quantization_applied": self.quantization_applied,
                    "onnx_enabled": self.onnx_session is not None,
                    "async_logging_used": timings.get("async_logging_used", 0) > 0,
                    "onnx_predict_calls": int(timings.get("onnx_predict_calls", 0)),
                    "pytorch_predict_calls": int(timings.get("pytorch_predict_calls", 0)),
                }
                print({
                    "service": "inference",
                    "event": "predict_timing_breakdown",
                    **timing_breakdown
                }, flush=True)
            except Exception as _te:
                print({"service": "inference", "event": "timing_breakdown_fail", "error": str(_te)})
            
            # Emit predict_inference_done with comprehensive metadata
            try:
                print({
                    "service": "inference",
                    "event": "predict_inference_done",
                    "inference_id": getattr(self, "current_run_id", "unknown"),
                    "duration_ms": round(timings.get("overall_ms", 0.0), 3),
                    "model_type": self.model_type or "unknown",
                    "run_id": getattr(self, "current_run_id", "unknown"),
                    "prediction_steps": int(df_transformed_predictions.shape[0]),
                    "input_sequence_length": int(self.input_seq_len),
                    "output_shape": list(df_transformed_predictions.shape),
                    "model_class": self.model_class,
                }, flush=True)
            except Exception as _de:
                print({"service": "inference", "event": "predict_done_log_fail", "error": str(_de)})
            
            try:
                print({
                    "service": "inference",
                    "event": "inference_stage_timings",
                    "timings_ms": {k: round(v, 3) for k, v in timings.items() if isinstance(v, (int, float))},
                    "timing_metadata": {k: v for k, v in timings.items() if not isinstance(v, (int, float))},
                    "model_class": self.model_class,
                    "rows_in": int(total_rows),
                    "rows_out": int(df_transformed_predictions.shape[0])
                })
            except Exception:
                pass
            return df_transformed_predictions
        finally:
            _finalize_timings()
            self._mark_job_finished()

    def _perform_pytorch_inference(self, df_eval: pd.DataFrame, df_predictions: pd.DataFrame, local_inference_length: int, timings: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """PyTorch inference logic"""
        import torch

        FEATURES = df_eval.columns.difference(TIME_FEATURES, sort=False).tolist()
        # Heuristic target column resolution (training created synthetic 'value' from fallback 'down')
        target_col = 'value' if 'value' in df_eval.columns else 'down'
        if target_col not in df_predictions.columns:
            # Ensure target column exists to receive predictions
            df_predictions[target_col] = np.nan

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Phase-5: Window preparation timing
        window_start = time.perf_counter()
        X_eval, _ = window_data(df_eval, TIME_FEATURES, self.input_seq_len, self.output_seq_len)
        if timings is not None:
            timings["window_data_ms"] = (time.perf_counter() - window_start) * 1000.0
            timings["window_preparation_ms"] = timings["window_data_ms"]  # Alias for Phase-5 reporting
            timings["window_rows"] = float(X_eval.shape[0])
        X_eval_tensor = torch.from_numpy(X_eval).float().to(device)

        remaining_real_data = X_eval.shape[0] - SAMPLE_IDX
        available_future_steps = min(remaining_real_data, local_inference_length)

        progress_interval = int(os.environ.get("PREDICT_PROGRESS_INTERVAL", 25))
        per_step_errors = []  # will store (step, mae_over_features) for overlapping steps with real data

        def _timed_predict(payload):
            start = time.perf_counter()
            # Phase-2: Try ONNX Runtime first if available, fallback to PyTorch
            if self.onnx_session is not None:
                try:
                    import torch
                    # ONNX expects numpy array input
                    if isinstance(payload, torch.Tensor):
                        onnx_input = payload.cpu().numpy()
                    else:
                        onnx_input = payload
                    
                    # Ensure correct shape for ONNX (batch, seq_len, features)
                    if onnx_input.ndim == 2:
                        onnx_input = np.expand_dims(onnx_input, axis=0)
                    
                    ort_inputs = {self.onnx_session.get_inputs()[0].name: onnx_input.astype(np.float32)}
                    ort_outputs = self.onnx_session.run(None, ort_inputs)
                    result = ort_outputs[0]  # ONNX returns list of outputs
                    
                    if timings is not None:
                        timings.setdefault("model_predict_ms", 0.0)
                        timings.setdefault("model_predict_calls", 0.0)
                        timings.setdefault("onnx_predict_calls", 0.0)
                        timings["model_predict_ms"] += (time.perf_counter() - start) * 1000.0
                        timings["model_predict_calls"] += 1.0
                        timings["onnx_predict_calls"] += 1.0
                    return result
                except Exception as onnx_err:
                    # ONNX failed, fallback to PyTorch
                    print({"service": "inference", "event": "onnx_predict_fallback", "error": str(onnx_err)[:200]})
                    self.onnx_session = None  # Disable for subsequent calls
            
            # Standard PyTorch/MLflow prediction
            result = self.current_model.predict(payload)  # type: ignore
            if timings is not None:
                timings.setdefault("model_predict_ms", 0.0)
                timings.setdefault("model_predict_calls", 0.0)
                timings.setdefault("pytorch_predict_calls", 0.0)
                timings["model_predict_ms"] += (time.perf_counter() - start) * 1000.0
                timings["model_predict_calls"] += 1.0
                timings["pytorch_predict_calls"] += 1.0
            return result

        loop_start = None
        with torch.no_grad():
            current_sequence = X_eval_tensor[SAMPLE_IDX].unsqueeze(0).to(device)
            if timings is not None:
                loop_start = time.perf_counter()

            for step in range(local_inference_length):
                # Predict a full block of up to output_seq_len steps
                # The MLflow pyfunc.predict implementation often expects 2-D inputs (n_samples, n_features).
                # Many torch pipelines build 3-D tensors (1, seq_len, n_features). Wrap predict with
                # tolerant fallbacks: try as-is, then squeeze leading batch dim, then flatten to 2-D.
                data_np = current_sequence.cpu().numpy()
                try:
                    multi_step_pred = _timed_predict(data_np)
                except Exception as e_pred:
                    # Log the original error and attempt fallbacks
                    print({
                        "service": "inference",
                        "event": "pyfunc_predict_error",
                        "error": str(e_pred),
                        "shape": getattr(data_np, "shape", None),
                    })
                    multi_step_pred = None
                    # Strategy 1: squeeze leading singleton batch dim -> (seq_len, n_features)
                    try:
                        if data_np.ndim == 3 and data_np.shape[0] == 1:
                            alt = data_np.squeeze(0)
                            multi_step_pred = _timed_predict(alt)
                            print({"service": "inference", "event": "pyfunc_predict_fallback_squeeze", "orig_shape": data_np.shape, "new_shape": getattr(alt, "shape", None)})
                    except Exception as e2:
                        print({"service": "inference", "event": "pyfunc_predict_fallback_squeeze_fail", "error": str(e2)})
                    # Strategy 2: flatten all timesteps into single feature vector -> (1, seq_len*n_features)
                    if multi_step_pred is None:
                        try:
                            flat = data_np.reshape(1, -1)
                            multi_step_pred = _timed_predict(flat)
                            print({"service": "inference", "event": "pyfunc_predict_fallback_flatten", "orig_shape": data_np.shape, "new_shape": getattr(flat, "shape", None)})
                        except Exception as e3:
                            print({"service": "inference", "event": "pyfunc_predict_fallback_flatten_fail", "error": str(e3)})
                    # If still None, raise original exception to be handled upstream
                    if multi_step_pred is None:
                        raise
                steps_to_use = min(self.output_seq_len, local_inference_length - step)

                for i in range(steps_to_use):
                    absolute_step = step + i
                    if absolute_step >= local_inference_length:
                        break

                    current_pred = multi_step_pred[:, i, :].flatten()
                    # Shape alignment: model may output only target (dim=1) or full feature vector.
                    if current_pred.shape[0] == len(FEATURES):
                        df_predictions.loc[df_predictions.index[absolute_step], FEATURES] = current_pred
                    elif current_pred.shape[0] == 1:
                        df_predictions.loc[df_predictions.index[absolute_step], target_col] = float(current_pred.item())
                    else:
                        print({
                            "service": "inference",
                            "event": "unexpected_pred_dim",
                            "pred_dim": int(current_pred.shape[0]),
                            "n_features": int(len(FEATURES))
                        })

                    # Phase-5: Optimized autoregressive loop - reduce allocations
                    next_step_idx = SAMPLE_IDX + absolute_step + 1
                    if next_step_idx < X_eval_tensor.shape[0]:
                        # Safe: use the next real row (in-place update to reuse memory)
                        current_sequence = X_eval_tensor[next_step_idx].unsqueeze(0).to(device)
                    else:
                        # Need to extend with predictions (recursive mode). We may only have endogenous prediction(s).
                        extension_idx = absolute_step + 1 - available_future_steps
                        if extension_idx < df_predictions.shape[0]:
                            # Phase-5: Reduce allocations by reusing current_sequence buffer
                            # Build a full feature vector of the SAME dimensionality as the original input sequence.
                            feature_dim = current_sequence.shape[-1]
                            pred_dim = current_pred.shape[0]
                            
                            # Preallocate once at loop start (amortized cost)
                            if not hasattr(self, '_pred_buffer_cache'):
                                self._pred_buffer_cache = {}
                            
                            cache_key = (feature_dim, pred_dim)
                            if cache_key not in self._pred_buffer_cache:
                                self._pred_buffer_cache[cache_key] = torch.zeros(1, 1, feature_dim, device=device)
                            
                            pred_tensor_full = self._pred_buffer_cache[cache_key]
                            
                            # Fill with prediction (in-place)
                            pred_tensor_full[0, 0, :pred_dim] = torch.from_numpy(current_pred).to(device)
                            
                            # Fill remaining exogenous feature slots with last known real values (persistence strategy)
                            if pred_dim < feature_dim:
                                pred_tensor_full[0, 0, pred_dim:] = current_sequence[0, -1, pred_dim:]
                            # Log when dimensionality repair occurs
                            if pred_dim != feature_dim:
                                print({
                                    "service": "inference",
                                    "event": "recursive_extension_pad",
                                    "pred_dim": int(pred_dim),
                                    "feature_dim": int(feature_dim),
                                    "strategy": "pad_with_last_exogenous"
                                })
                            # Phase-5: In-place concatenation using slice assignment
                            current_sequence[:, :-1, :] = current_sequence[:, 1:, :].clone()
                            current_sequence[:, -1:, :] = pred_tensor_full
                        else:
                            print(f"[Warning] df_predictions extension exhausted at index {extension_idx}. Stopping inference.")
                            break

                # Compute per-step error if within available_future_steps
                if step < available_future_steps:
                    try:
                        actual_idx = SAMPLE_IDX + self.input_seq_len + step
                        if actual_idx < df_eval.shape[0]:
                            if current_pred.shape[0] == len(FEATURES):
                                compare_cols = FEATURES
                            else:
                                compare_cols = [c for c in [target_col] if c in df_eval.columns and c in df_predictions.columns]
                            if compare_cols:
                                actual_row = df_eval.iloc[actual_idx][compare_cols]
                                pred_row = df_predictions.iloc[step][compare_cols]
                                mae_step = float(np.mean(np.abs(pred_row.values - actual_row.values)))
                                per_step_errors.append({"step": step, "mae": mae_step})
                    except Exception:
                        pass

                if progress_interval > 0 and (step + 1) % progress_interval == 0:
                    print({
                        "service": "inference",
                        "event": "progress",
                        "step": step + 1,
                        "total": local_inference_length,
                        "pct": round(100.0 * (step + 1) / local_inference_length, 2)
                    })

                step += steps_to_use - 1  # outer loop also increments


        if timings is not None and loop_start is not None:
            timings["pytorch_loop_ms"] = (time.perf_counter() - loop_start) * 1000.0

        df_predictions = df_predictions.drop(columns=TIME_FEATURES)

        if self.current_scaler is not None:
            inv_start = time.perf_counter()
            try:
                original_cols = (
                    list(getattr(self.current_scaler, "feature_names_in_", []))
                    or list(getattr(self.current_scaler, "feature_names", []))
                )
                if original_cols:
                    sub_scaler = subset_scaler(self.current_scaler, original_cols, df_predictions.columns.tolist())
                else:
                    # Fallback: scaler may not have feature names stored (older sklearn); use length check
                    sub_scaler = self.current_scaler
                inv = sub_scaler.inverse_transform(df_predictions)
                df_transformed_predictions = pd.DataFrame(inv, index=df_predictions.index, columns=df_predictions.columns)
            except Exception as e:  # noqa: BLE001
                print(f"[Warning] inverse scaling failed ({e}); returning raw predictions.")
                df_transformed_predictions = df_predictions.copy()
            finally:
                if timings is not None:
                    timings.setdefault("inverse_scale_ms", 0.0)
                    timings["inverse_scale_ms"] += (time.perf_counter() - inv_start) * 1000.0
        else:
            print("[Warning] current_scaler is None. Returning raw predictions.")
            df_transformed_predictions = df_predictions.copy()
            if timings is not None:
                timings.setdefault("inverse_scale_ms", 0.0)

        print(f"PyTorch Inference completed:")
        print(f"- Used actual future values for first {min(available_future_steps, local_inference_length)} steps")
        if local_inference_length > available_future_steps:
            print(f"- Switched to recursive mode after step {available_future_steps}")
        print(f"- Model predicts {self.output_seq_len} step(s) at a time")
        print(f"- Total predictions generated: {df_transformed_predictions.shape[0]}")
        try:
            print({
                "service": "inference",
                "event": "inference_shape_summary",
                "feature_dim": int(X_eval_tensor.shape[-1]),
                "output_seq_len": int(self.output_seq_len),
                "input_seq_len": int(self.input_seq_len),
                "pred_columns": df_transformed_predictions.columns.tolist()
            })
        except Exception:
            pass

        # Attach per-step error list to instance for later logging in save method (trim long list)
        try:
            max_err_steps = int(os.environ.get("PREDICT_MAX_ERROR_STEPS", 200))
            self._last_per_step_errors = per_step_errors[:max_err_steps]
        except Exception:
            self._last_per_step_errors = per_step_errors

        return df_transformed_predictions

    def _perform_prophet_inference(self, df_eval: pd.DataFrame, df_predictions: pd.DataFrame, local_inference_length: int, timings: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Prophet inference logic"""
        predict_start = time.perf_counter()
        # Get predictions from Prophet model
        df_predictions = self.current_model.predict(df_predictions) # type: ignore
        if timings is not None:
            timings.setdefault("model_predict_ms", 0.0)
            timings.setdefault("model_predict_calls", 0.0)
            timings["model_predict_ms"] += (time.perf_counter() - predict_start) * 1000.0
            timings["model_predict_calls"] += 1.0
        
        # Apply inverse scaling if scaler is available
        if self.current_scaler is not None:
            inv_start = time.perf_counter()
            df_transformed_predictions = pd.DataFrame(
                self.current_scaler.inverse_transform(df_predictions),
                index=df_predictions.index,
                columns=df_predictions.columns
            )
            if timings is not None:
                timings.setdefault("inverse_scale_ms", 0.0)
                timings["inverse_scale_ms"] += (time.perf_counter() - inv_start) * 1000.0
        else:
            print("[Warning] current_scaler is None. Returning raw predictions.")
            df_transformed_predictions = df_predictions.copy()
            if timings is not None:
                timings.setdefault("inverse_scale_ms", 0.0)

        # Prophet inference summaries disabled
        pass

        return df_transformed_predictions

    def _perform_statsforecast_inference(self, df_eval: pd.DataFrame, df_predictions: pd.DataFrame, local_inference_length: int, timings: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """StatsForecast inference logic"""

        if self.params["downsampling"] == "0" or self.params["downsampling"] == self.params["frequency"]:
            exog_df = df_predictions[TIME_FEATURES] if TIME_FEATURES else None

            input_dict = {
                "h": INFERENCE_LENGTH,
                "X": exog_df,
                "level": None
            }
        else:
            downsampling = pd.Timedelta(self.params["downsampling"])
            frequency = pd.Timedelta(self.params["frequency"])
            inf_len: int = int(np.ceil(local_inference_length*frequency/downsampling))

            if TIME_FEATURES:
                df_predictions = pd.DataFrame(
                    index=pd.date_range(
                        start=df_eval.index[SAMPLE_IDX],
                        periods=inf_len,
                        freq=frequency
                    ),
                    columns=df_eval.columns
                )

                df_predictions = time_to_feature(df_predictions)
                exog_df = df_predictions[TIME_FEATURES]
            else:
                exog_df = None

            input_dict = {
                "h": inf_len,
                "X": exog_df,
                "level": None
            }
        predict_start = time.perf_counter()
        df_predictions = self.current_model.predict(input_dict) # type: ignore
        if timings is not None:
            timings.setdefault("model_predict_ms", 0.0)
            timings.setdefault("model_predict_calls", 0.0)
            timings["model_predict_ms"] += (time.perf_counter() - predict_start) * 1000.0
            timings["model_predict_calls"] += 1.0
        
        # Apply inverse scaling if scaler is available
        if self.current_scaler is not None:
            inv_start = time.perf_counter()
            df_transformed_predictions = pd.DataFrame(
                self.current_scaler.inverse_transform(df_predictions),
                index=df_predictions.index,
                columns=df_predictions.columns
            )
            if timings is not None:
                timings.setdefault("inverse_scale_ms", 0.0)
                timings["inverse_scale_ms"] += (time.perf_counter() - inv_start) * 1000.0
        else:
            print("[Warning] current_scaler is None. Returning raw predictions.")
            df_transformed_predictions = df_predictions.copy()
            if timings is not None:
                timings.setdefault("inverse_scale_ms", 0.0)

        # StatsForecast inference summaries disabled
        pass

        return df_transformed_predictions

    def _save_and_publish_predictions(
        self,
        df_transformed_predictions: pd.DataFrame,
        df_eval: Optional[pd.DataFrame] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> None:
        """Write a single JSON object (one line) per inference batch to MinIO (JSONL) and emit Kafka confirmation.

        Required JSON fields:
          - timestamp (UTC ISO8601 Z)
          - identifier
          - run_id
          - model_type
          - config_hash
          - status (SUCCESS/FAILURE)
          - metrics (dict: mae_mean, mse_mean, rmse, per-feature errors, step_mae, etc.)
          - samples (list of sample prediction dicts)
        Storage layout: bucket=inference-logs (or INFERENCE_LOG_BUCKET env override), object key: {identifier}/{YYYYMMDD}/results.jsonl
        Append only: fetch existing object (if any), add one new line, re-upload.
        """
        if os.getenv("INFERENCE_DISABLE_LOG_UPLOAD", "0") in {"1", "true", "TRUE"}:
            if timings is not None:
                timings.setdefault("log_upload_skipped", 0.0)
                timings["log_upload_skipped"] += 1.0
            return

        from datetime import datetime
        import json, math, hashlib
        from client_utils import get_file

        identifier = os.environ.get("IDENTIFIER", "default") or "default"
        bucket = os.environ.get("INFERENCE_LOG_BUCKET", "inference-logs")
        date_part = datetime.utcnow().strftime("%Y%m%d")
        object_key = f"{identifier}/{date_part}/results.jsonl"
        status = "SUCCESS"

        # --- Metrics & Samples -------------------------------------------------
        metrics_block: dict = {}
        samples_block: list = []
        metrics_start = time.perf_counter()
        try:
            if df_eval is not None and not df_eval.empty:
                pred_idx = df_transformed_predictions.index
                overlap_idx = pred_idx.intersection(df_eval.index)
                if len(overlap_idx) > 0:
                    predicted_cols = df_transformed_predictions.columns.tolist()
                    actual_subset = df_eval.loc[overlap_idx, predicted_cols].copy()
                    # Try inverse scaling to original scale for actuals
                    if self.current_scaler is not None:
                        try:
                            original_cols = list(getattr(self.current_scaler, "feature_names_in_", [])) or []
                            if original_cols:
                                sub_scaler = subset_scaler(self.current_scaler, original_cols, predicted_cols)
                                actual_inv = sub_scaler.inverse_transform(actual_subset)
                                actual_subset_df = pd.DataFrame(actual_inv, index=actual_subset.index, columns=predicted_cols)
                            else:
                                actual_subset_df = actual_subset
                        except Exception:
                            actual_subset_df = actual_subset
                    else:
                        actual_subset_df = actual_subset
                    preds_overlap = df_transformed_predictions.loc[overlap_idx]
                    err_mae = (preds_overlap - actual_subset_df).abs().mean().to_dict()
                    err_mse = ((preds_overlap - actual_subset_df)**2).mean().to_dict()
                    # Collect non-NaN values for aggregate means
                    mae_vals = [float(v) for v in err_mae.values() if v is not None and not math.isnan(float(v))]
                    mse_vals = [float(v) for v in err_mse.values() if v is not None and not math.isnan(float(v))]
                    mae_mean = float(np.mean(mae_vals)) if mae_vals else math.nan
                    mse_mean = float(np.mean(mse_vals)) if mse_vals else math.nan
                    rmse = float(math.sqrt(mse_mean)) if (not math.isnan(mse_mean)) else math.nan
                    metrics_block = {
                        "overlap_rows": int(len(overlap_idx)),
                        "mae_mean": mae_mean,
                        "mse_mean": mse_mean,
                        "rmse": rmse,
                        # Filter out features that are entirely NaN so logs stay cleaner
                        "mae": {k: float(v) for k, v in err_mae.items() if v is not None and not math.isnan(float(v))},
                        "mse": {k: float(v) for k, v in err_mse.items() if v is not None and not math.isnan(float(v))},
                    }
                    # Samples (bounded)
                    feature_limit = int(os.environ.get("PREDICT_LOG_FEATURE_LIMIT", 3))
                    feats_for_samples = predicted_cols[:feature_limit]
                    n_pred = len(df_transformed_predictions)
                    sample_positions = sorted({0, n_pred-1, n_pred//10, n_pred//2, (9*n_pred)//10})
                    for pos in sample_positions:
                        if 0 <= pos < n_pred:
                            ts = df_transformed_predictions.index[pos]
                            row_pred = df_transformed_predictions.iloc[pos][feats_for_samples].to_dict()
                            if ts in actual_subset_df.index:
                                row_actual = actual_subset_df.loc[ts][feats_for_samples].to_dict()
                                row_err = {f: float(abs(row_pred[f]-row_actual[f])) for f in feats_for_samples}
                            else:
                                row_actual = None
                                row_err = {}
                            samples_block.append({
                                "step": int(pos),
                                "ts": ts.isoformat(),
                                "pred": {k: float(v) for k, v in row_pred.items()},
                                "actual": ({k: float(v) for k, v in row_actual.items()} if row_actual else None),
                                "abs_err": row_err
                            })
        except Exception as e:  # noqa: BLE001
            metrics_block = {"metrics_error": str(e)}
        finally:
            if timings is not None:
                timings.setdefault("metrics_block_ms", 0.0)
                timings["metrics_block_ms"] += (time.perf_counter() - metrics_start) * 1000.0
                timings.setdefault("samples_count", 0.0)
                timings["samples_count"] += float(len(samples_block))

        # Step MAE sequence if available
        if hasattr(self, "_last_per_step_errors") and getattr(self, "_last_per_step_errors"):
            metrics_block["step_mae"] = self._last_per_step_errors

        # Prediction hash for quick diff / lineage
        try:
            pred_hash = hashlib.sha256(str(df_transformed_predictions.head(3).to_dict()).encode()).hexdigest()[:16]
        except Exception:
            pred_hash = ""
        metrics_block["prediction_hash"] = pred_hash
        metrics_block["rows_predicted"] = int(len(df_transformed_predictions))

        # Deduplication: skip if we've already emitted this exact prediction hash for this run
        run_id = getattr(self, "current_run_id", "")
        pred_key = (run_id, pred_hash)
        with self._emitted_prediction_lock:
            if run_id and pred_hash and pred_key in self._emitted_prediction_keys:
                print({
                    "service": "inference",
                    "event": "duplicate_prediction_skip",
                    "run_id": run_id,
                    "prediction_hash": pred_hash
                })
                if timings is not None:
                    timings.setdefault("save_publish_dedup_skips", 0.0)
                    timings["save_publish_dedup_skips"] += 1.0
                return  # Do not append another identical line
            if run_id and pred_hash:
                self._emitted_prediction_keys.add(pred_key)

        # Build JSON line
        serialize_start = time.perf_counter()
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "identifier": identifier,
            "run_id": getattr(self, "current_run_id", ""),
            "model_type": self.model_type or getattr(self, "current_run_name", ""),
            "config_hash": getattr(self, "current_config_hash", None),
            "status": status,
            "metrics": metrics_block,
            "samples": samples_block,
        }

        line = json.dumps(record, default=str) + "\n"
        if timings is not None:
            timings.setdefault("json_serialize_ms", 0.0)
            timings["json_serialize_ms"] += (time.perf_counter() - serialize_start) * 1000.0

        # --- Append to MinIO object (download + append + re-upload) -----------
        from client_utils import post_file
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                fetch_start = time.perf_counter()
                try:
                    existing_obj = get_file(self.gateway_url, bucket, object_key)
                except Exception:
                    existing_obj = None
                finally:
                    if timings is not None:
                        timings.setdefault("log_fetch_ms", 0.0)
                        timings["log_fetch_ms"] += (time.perf_counter() - fetch_start) * 1000.0
                        timings.setdefault("log_fetch_calls", 0.0)
                        timings["log_fetch_calls"] += 1.0
                if existing_obj is None:
                    existing_bytes = b""
                else:
                    try:
                        existing_bytes = existing_obj.getvalue()  # type: ignore[attr-defined]
                    except Exception:
                        existing_bytes = existing_obj if isinstance(existing_obj, (bytes, bytearray)) else b""
                new_body = existing_bytes + line.encode()
                upload_start = time.perf_counter()
                post_file(self.gateway_url, bucket, object_key, new_body)
                if timings is not None:
                    timings.setdefault("log_upload_ms", 0.0)
                    timings["log_upload_ms"] += (time.perf_counter() - upload_start) * 1000.0
                    timings.setdefault("log_upload_attempts", 0.0)
                    timings["log_upload_attempts"] += 1.0
                print({
                    "service": "inference",
                    "event": "inference_log_write",
                    "bucket": bucket,
                    "object_key": object_key,
                    "identifier": identifier,
                    "lines_appended": 1,
                    "bytes_appended": len(line)
                })
                break
            except Exception as e:  # noqa: BLE001
                if timings is not None:
                    timings.setdefault("log_upload_failures", 0.0)
                    timings["log_upload_failures"] += 1.0
                if attempt == max_retries:
                    publish_error(
                        self.producer,
                        dlq_topic=os.environ.get("DLQ_PERFORMANCE_TOPIC", "DLQ-performance-eval"),
                        operation="Inference Log Write",
                        status="Failure",
                        error_details=str(e),
                        payload={"object_key": object_key, "identifier": identifier, "attempts": attempt},
                    )
                else:
                    print(f"[Warning] inference JSONL log write attempt {attempt} failed: {e}")

        # --- Publish Kafka success event --------------------------------------
        try:
            publish_start = time.perf_counter()
            produce_message(self.producer, self.output_topic, {
                "operation": "Inference",
                "status": status,
                "identifier": identifier,
                "log_bucket": bucket,
                "log_object_key": object_key,
                "run_id": record.get("run_id"),
                "model_type": record.get("model_type"),
                "config_hash": record.get("config_hash"),
                "rows": metrics_block.get("rows_predicted", 0)
            })
            if timings is not None:
                timings.setdefault("kafka_publish_ms", 0.0)
                timings["kafka_publish_ms"] += (time.perf_counter() - publish_start) * 1000.0
                timings.setdefault("kafka_publish_calls", 0.0)
                timings["kafka_publish_calls"] += 1.0
        except Exception as e:  # noqa: BLE001
            print(f"Kafka inference publish error (non-fatal): {e}")
            if timings is not None:
                timings.setdefault("kafka_publish_errors", 0.0)
                timings["kafka_publish_errors"] += 1.0