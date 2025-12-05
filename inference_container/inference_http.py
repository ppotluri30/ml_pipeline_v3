"""
HTTP-only inference server entrypoint.

Loads promoted models via current.json pointer and serves predictions via FastAPI.
NO Kafka consumers - pure HTTP service for /predict, /healthz, /metrics.

This module creates a global inferencer instance that api_server.py will import.
"""
import os
import sys
import time
import json as _json

# Ensure inference module utilities are available
sys.path.insert(0, os.path.dirname(__file__))

from inferencer import Inferencer
from client_utils import get_file


def _convert_model_to_onnx(service: Inferencer, run_id: str) -> None:
    """Convert PyTorch model to ONNX Runtime for faster inference."""
    import os
    
    _ENABLE_ONNX = os.environ.get("INFERENCE_ENABLE_ONNX", "1").lower() in {"1", "true", "yes"}
    if not _ENABLE_ONNX:
        _log("onnx_disabled_by_env", run_id=run_id)
        return
    
    model = service.current_model
    if model is None or not hasattr(model, "_model_impl"):
        _log("onnx_skip_no_model_impl", run_id=run_id)
        return
    
    try:
        import torch
        import tempfile
        import hashlib
        
        # Try importing onnxruntime and onnx
        try:
            import onnxruntime as ort
            import onnx
        except ImportError as ie:
            _log("onnx_skip_not_installed", run_id=run_id, error=str(ie))
            return
        
        # Extract PyTorch model from MLflow wrapper
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
            _log("onnx_skip_no_pytorch_model", run_id=run_id, 
                 has_model_impl=hasattr(model, "_model_impl"),
                 impl_attrs=dir(model._model_impl) if hasattr(model, "_model_impl") else [])
            return
        
        inner_model.eval()
        
        # Create cache path based on run_id
        cache_dir = os.path.join(tempfile.gettempdir(), "onnx_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_str = str(inner_model)
        cache_key = hashlib.md5(f"{run_id}_{model_str}".encode()).hexdigest()[:16]
        onnx_cache_path = os.path.join(cache_dir, f"{cache_key}.onnx")
        
        # Check if ONNX model already exists and is valid
        if os.path.exists(onnx_cache_path):
            try:
                onnx_model = onnx.load(onnx_cache_path)
                onnx.checker.check_model(onnx_model)
                
                # Load into ONNX Runtime with optimized settings
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 2
                sess_options.inter_op_num_threads = 2
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                service.onnx_session = ort.InferenceSession(onnx_cache_path, sess_options)
                service.onnx_cache_path = onnx_cache_path
                service.model_backend = "onnx"
                
                _log("onnx_loaded_from_cache", run_id=run_id, cache_path=onnx_cache_path)
                return
            except Exception as cache_err:
                _log("onnx_cache_invalid_regenerating", run_id=run_id, error=str(cache_err)[:200])
                try:
                    os.unlink(onnx_cache_path)
                except Exception:
                    pass
        
        # Create dummy input for export
        input_seq_len = service.input_seq_len or 10
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
        
        # Export to ONNX
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
        
        # Load into ONNX Runtime with optimized settings
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        service.onnx_session = ort.InferenceSession(onnx_cache_path, sess_options)
        service.onnx_cache_path = onnx_cache_path
        service.model_backend = "onnx"
        
        _log("onnx_export_success", run_id=run_id, model_type=service.model_type, 
             input_shape=[1, input_seq_len, feature_dim], cache_path=onnx_cache_path)
    except Exception as onnx_err:
        service.onnx_session = None
        service.model_backend = "pytorch"
        _log("onnx_conversion_failed", run_id=run_id, error=str(onnx_err)[:300])


# Environment variables
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://fastapi-app:8000")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
IDENTIFIER = os.environ.get("IDENTIFIER", "default")


def _log(event: str, **kwargs):
    """Structured logging helper."""
    log_entry = {"service": "inference_http", "event": event}
    log_entry.update(kwargs)
    print(_json.dumps(log_entry))


# Global state for auto-reload mechanism
_last_loaded_run_id = None
_reload_check_interval = int(os.environ.get("MODEL_RELOAD_CHECK_INTERVAL", "30"))  # seconds
_reload_thread = None
_reload_shutdown = False

# Warm-up readiness flag - only True after model loaded AND warm-up inference complete
_warmup_ready = False
_warmup_error: str | None = None


def is_warmup_ready() -> bool:
    """Check if warm-up is complete and service is ready for traffic."""
    return _warmup_ready


def get_warmup_status() -> dict:
    """Get detailed warm-up status for readiness probes."""
    return {
        "warmup_ready": _warmup_ready,
        "warmup_error": _warmup_error,
        "model_loaded": inferencer.current_model is not None if 'inferencer' in globals() else False,
        "run_id": inferencer.current_run_id if 'inferencer' in globals() else None,
        "model_type": inferencer.model_type if 'inferencer' in globals() else None,
    }


def _get_current_pointer_run_id() -> tuple:
    """
    Check current.json pointer and return (run_id, pointer_dict).
    Returns (None, None) if pointer not found or invalid.
    """
    GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://fastapi-app:8000")
    IDENTIFIER = os.environ.get("IDENTIFIER", "default")
    
    pointer_paths = [
        ("model-promotion", "current.json"),
        ("model-promotion", "global/current.json"),
        ("model-promotion", f"{IDENTIFIER}/current.json"),
    ]
    
    for bucket, object_key in pointer_paths:
        try:
            file_stream = get_file(GATEWAY_URL, bucket, object_key)
            if not file_stream:
                continue
            
            content = file_stream.read().decode('utf-8')
            pointer = _json.loads(content)
            run_id = pointer.get("run_id")
            
            if run_id:
                return (run_id, pointer)
        except Exception:
            continue
    
    return (None, None)


def _load_promoted_pointer(service: Inferencer) -> bool:
    """
    Load the promoted model from current.json pointer.
    Returns True if successfully loaded, False otherwise.
    """
    GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://fastapi-app:8000")
    IDENTIFIER = os.environ.get("IDENTIFIER", "default")
    
    # Try multiple pointer locations (root, global, identifier-scoped)
    pointer_paths = [
        ("model-promotion", "current.json"),
        ("model-promotion", "global/current.json"),
        ("model-promotion", f"{IDENTIFIER}/current.json"),
    ]
    
    for bucket, object_key in pointer_paths:
        try:
            path_str = f"{bucket}/{object_key}"
            url = f"{GATEWAY_URL}/download/{path_str}"
            _log("promotion_pointer_fetch_attempt", url=url, path=path_str)
            
            file_stream = get_file(GATEWAY_URL, bucket, object_key)
            if not file_stream:
                continue
            
            # Read from BytesIO stream
            content = file_stream.read().decode('utf-8')
            pointer = _json.loads(content)
            run_id = pointer.get("run_id")
            model_uri = pointer.get("model_uri")
            model_type = pointer.get("model_type")
            
            if not run_id or not model_uri:
                _log("promotion_pointer_invalid", path=path_str, pointer=pointer)
                continue
            
            _log("promotion_pointer_parsed", run_id=run_id, model_uri=model_uri, 
                 model_type=model_type, path=path_str)
            
            # Load the model directly using MLflow pyfunc
            from mlflow import pyfunc
            uri_candidates = [model_uri]
            if not model_uri.rstrip('/').endswith('/model'):
                uri_candidates.append(model_uri.rstrip('/') + '/model')
            
            loaded = False
            for cand in uri_candidates:
                try:
                    service.current_model = pyfunc.load_model(cand)
                    service.current_run_id = run_id
                    service.model_type = model_type or ''
                    service.current_run_name = model_type or ''
                    service.current_experiment_name = pointer.get("experiment", "Default")
                    service.current_config_hash = pointer.get("config_hash")
                    
                    # Set model_class based on model_type
                    upper_type = (model_type or '').upper()
                    if upper_type in ('GRU', 'LSTM'):
                        service.model_class = 'pytorch'
                    elif upper_type in ('PROPHET',):
                        service.model_class = 'prophet'
                    elif upper_type in ('STATSFORECAST', 'ARIMA', 'SARIMAX'):
                        service.model_class = 'statsforecast'
                    else:
                        # Default to pytorch for neural network models
                        service.model_class = 'pytorch'
                    
                    loaded = True
                    _log("model_loaded", run_id=run_id, model_uri=cand, model_type=model_type, model_class=service.model_class)
                    break
                except Exception as load_err:
                    _log("model_load_attempt_failed", candidate=cand, error=str(load_err))
                    continue
            
            if not loaded:
                _log("model_load_failed_all_candidates", run_id=run_id, model_uri=model_uri)
                continue
            
            # Enrich metadata from MLflow
            try:
                import mlflow
                run = mlflow.get_run(run_id)
                params = run.data.params or {}
                
                if 'input_sequence_length' in params:
                    try: 
                        service.input_seq_len = int(params['input_sequence_length'])
                    except: 
                        pass
                if 'output_sequence_length' in params:
                    try: 
                        service.output_seq_len = int(params['output_sequence_length'])
                    except: 
                        pass
                        
                _log("promotion_model_enriched", run_id=run_id, model_type=service.model_type,
                     input_seq_len=service.input_seq_len, output_seq_len=service.output_seq_len)
            except Exception as enrich_err:
                _log("promotion_model_enrich_fail", run_id=run_id, error=str(enrich_err))
            
            # Convert PyTorch model to ONNX for faster inference
            if service.model_class == 'pytorch':
                try:
                    _convert_model_to_onnx(service, run_id)
                except Exception as onnx_err:
                    _log("onnx_conversion_error", run_id=run_id, error=str(onnx_err)[:200])
            
            # Load scaler artifact (required for PyTorch models to inverse transform predictions)
            try:
                import tempfile
                import pickle
                from mlflow.artifacts import download_artifacts
                from mlflow.tracking import MlflowClient
                from data_utils import _fix_zero_scale
                
                client = MlflowClient()
                
                # Look for scaler .pkl file under 'scaler/' directory first
                try:
                    scaler_dir_items = client.list_artifacts(run_id, path="scaler")
                except Exception:
                    scaler_dir_items = []
                
                pkl_candidates = [a.path for a in scaler_dir_items 
                                  if not getattr(a, 'is_dir', False) and a.path.lower().endswith('.pkl')]
                
                # If nothing found in scaler/, look at root for any scaler-related pkl
                if not pkl_candidates:
                    try:
                        root_items = client.list_artifacts(run_id, path="")
                    except Exception:
                        root_items = []
                    root_pkls = [a.path for a in root_items 
                                 if not getattr(a, 'is_dir', False) and a.path.lower().endswith('.pkl')]
                    # Prefer names containing 'scaler'
                    root_pkls_sorted = sorted(root_pkls, key=lambda n: (0 if 'scaler' in n.lower() else 1, len(n)))
                    if root_pkls_sorted:
                        pkl_candidates = root_pkls_sorted
                
                # Rank candidates: contain 'scaler' first, then shorter name
                pkl_candidates = sorted(pkl_candidates, key=lambda n: (0 if 'scaler' in n.lower() else 1, len(n)))
                
                if pkl_candidates:
                    chosen_rel_path = pkl_candidates[0]
                    scaler_artifact_uri = f"runs:/{run_id}/{chosen_rel_path}"
                    try:
                        scaler_path = download_artifacts(artifact_uri=scaler_artifact_uri, dst_path=tempfile.gettempdir())
                        with open(scaler_path, "rb") as f:
                            service.current_scaler = pickle.load(f)
                        # Apply zero-scale fix to prevent division-by-zero during inverse_transform
                        if service.current_scaler is not None:
                            scaler_type_name = service.current_scaler.__class__.__name__
                            service.current_scaler = _fix_zero_scale(service.current_scaler, scaler_type_name=scaler_type_name)
                            _log("scaler_loaded", run_id=run_id, artifact_path=chosen_rel_path)
                    except Exception as scaler_err:
                        _log("scaler_load_failed", run_id=run_id, artifact_path=chosen_rel_path, error=str(scaler_err))
                else:
                    _log("scaler_not_found", run_id=run_id, note="No scaler .pkl located under scaler/ or root")
            except Exception as scaler_search_err:
                _log("scaler_search_error", run_id=run_id, error=str(scaler_search_err))
            
            _log("promotion_model_load_success", run_id=run_id, model_type=model_type, path=path_str)
            return True
            
        except Exception as e:
            _log("promotion_pointer_fetch_fail", path=path_str, error=str(e))
            continue
    
    _log("promotion_pointer_not_found", paths_tried=pointer_paths)
    return False


# Create global inferencer instance (api_server.py will import this)
# Initialize with required parameters (no Kafka producer needed for HTTP-only)
inferencer = Inferencer(
    gateway_url=GATEWAY_URL,
    producer=None,  # No Kafka producer in HTTP-only mode
    dlq_topic="",   # Not used in HTTP-only mode
    output_topic="" # Not used in HTTP-only mode
)


# Preload test dataframe for feature engineering warmup
def _preload_test_dataframe(service: Inferencer):
    """Preload a test DataFrame to warm up feature engineering utilities."""
    try:
        from data_utils import strip_timezones
        import pandas as pd
        
        test_df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=100, freq='2min'),
            'down': [1.0] * 100,
            'up': [1.0] * 100,
        })
        test_df, _ = strip_timezones(test_df)
        _log("preload_test_dataframe_success", rows=len(test_df))
    except Exception as e:
        _log("preload_test_dataframe_error", error=str(e))


def _execute_warmup_inference(service: Inferencer) -> bool:
    """
    Execute a warm-up inference to ensure model/ONNX runtime is fully initialized.
    Returns True if warm-up succeeded, False otherwise.
    """
    global _warmup_ready, _warmup_error
    
    try:
        if service.current_model is None:
            _log("warmup_skip", reason="no_model_loaded")
            return False
        
        import numpy as np
        
        input_seq_len = service.input_seq_len or 10
        
        _log("warmup_inference_start", model_type=service.model_type, 
             input_seq_len=input_seq_len, backend=getattr(service, 'model_backend', 'unknown'))
        
        t0 = time.time()
        
        # Warm-up ONNX session if available (preferred path)
        if getattr(service, 'onnx_session', None) is not None:
            # Get expected input shape from ONNX session
            input_info = service.onnx_session.get_inputs()[0]
            input_shape = input_info.shape  # e.g., ['batch', 'seq_len', 17]
            n_features = input_shape[-1] if isinstance(input_shape[-1], int) else 17
            
            # Create synthetic input matching expected shape
            warmup_input = np.random.randn(1, input_seq_len, n_features).astype(np.float32)
            
            # Run multiple warm-up passes to fully initialize ONNX runtime
            for i in range(3):
                _ = service.onnx_session.run(None, {input_info.name: warmup_input})
            
            elapsed_ms = int((time.time() - t0) * 1000)
            _log("warmup_inference_complete", elapsed_ms=elapsed_ms, 
                 model_type=service.model_type, backend="onnx", passes=3)
            return True
        
        # Fallback: warm-up PyTorch model directly
        elif getattr(service, 'model_class', '') == 'pytorch':
            import torch
            
            # Get the underlying model
            pyfunc_model = service.current_model
            if hasattr(pyfunc_model, '_model_impl'):
                inner = pyfunc_model._model_impl
                if hasattr(inner, 'python_model'):
                    inner = inner.python_model
                if hasattr(inner, 'model'):
                    torch_model = inner.model
                    
                    # Determine feature dimension from model architecture
                    n_features = 17  # default
                    if hasattr(torch_model, 'gru') and hasattr(torch_model.gru, 'input_size'):
                        n_features = torch_model.gru.input_size
                    elif hasattr(torch_model, 'lstm') and hasattr(torch_model.lstm, 'input_size'):
                        n_features = torch_model.lstm.input_size
                    
                    # Create synthetic input
                    warmup_input = torch.randn(1, input_seq_len, n_features)
                    
                    # Run inference passes
                    torch_model.eval()
                    with torch.no_grad():
                        for i in range(3):
                            _ = torch_model(warmup_input)
                    
                    elapsed_ms = int((time.time() - t0) * 1000)
                    _log("warmup_inference_complete", elapsed_ms=elapsed_ms,
                         model_type=service.model_type, backend="pytorch", passes=3)
                    return True
        
        # Skip warm-up for non-PyTorch models (Prophet, etc.)
        _log("warmup_skip", reason="no_compatible_backend", model_class=getattr(service, 'model_class', 'unknown'))
        return True  # Don't block readiness for Prophet models
        
    except Exception as e:
        _warmup_error = str(e)[:200]
        _log("warmup_inference_error", error=_warmup_error)
        return False


def _auto_reload_loop():
    """
    Background thread that periodically checks for model pointer changes
    and reloads the model automatically without pod restart.
    """
    global _last_loaded_run_id, _reload_shutdown
    
    _log("auto_reload_thread_started", check_interval_seconds=_reload_check_interval)
    
    while not _reload_shutdown:
        try:
            time.sleep(_reload_check_interval)
            
            if _reload_shutdown:
                break
            
            # Check if pointer has changed
            current_run_id, pointer = _get_current_pointer_run_id()
            
            if not current_run_id:
                _log("auto_reload_check", result="no_pointer_found")
                continue
            
            # Compare with last loaded run_id
            if current_run_id == _last_loaded_run_id:
                _log("auto_reload_check", result="no_change", run_id=current_run_id)
                continue
            
            # Pointer changed - reload model!
            _log("model_reload_detected", old_run_id=_last_loaded_run_id, 
                 new_run_id=current_run_id, pointer=pointer)
            
            # Perform atomic model reload
            reload_success = _load_promoted_pointer(inferencer)
            
            if reload_success:
                _last_loaded_run_id = current_run_id
                _log("model_reload_success", run_id=current_run_id, 
                     model_type=inferencer.model_type)
            else:
                _log("model_reload_error", run_id=current_run_id, 
                     error="Failed to load model from pointer")
                
        except Exception as e:
            _log("auto_reload_loop_error", error=str(e))
            time.sleep(5)  # Brief pause on error before retry
    
    _log("auto_reload_thread_stopped")


# Eagerly load promoted model at module import time (before FastAPI app starts)
# This ensures api_server.py sees a loaded model when it imports this module
_log("http_server_starting")
_preload_test_dataframe(inferencer)
model_loaded = _load_promoted_pointer(inferencer)
if model_loaded:
    _last_loaded_run_id = inferencer.current_run_id
    _log("initial_model_loaded", run_id=_last_loaded_run_id)
    
    # Execute warm-up inference before marking as ready
    warmup_success = _execute_warmup_inference(inferencer)
    if warmup_success:
        _warmup_ready = True
        _log("warmup_ready", run_id=_last_loaded_run_id, model_type=inferencer.model_type)
    else:
        # Model loaded but warm-up failed - still mark as ready but log warning
        _warmup_ready = True
        _log("warmup_failed_but_ready", run_id=_last_loaded_run_id, 
             error=_warmup_error, note="Proceeding despite warm-up failure")
else:
    _log("startup_warning", message="No promoted model loaded - will serve with empty model")
    # No model means we can't warm up, but we should still be "ready" for health checks
    # Kubernetes will route traffic only when /internal/ready returns 200
    _warmup_ready = False

# Start auto-reload background thread
import threading
_reload_thread = threading.Thread(target=_auto_reload_loop, daemon=True, name="model-auto-reload")
_reload_thread.start()
_log("auto_reload_enabled", check_interval_seconds=_reload_check_interval)


if __name__ == "__main__":
    # Start Uvicorn server (uses string path for multiprocess workers)
    import uvicorn
    
    UVICORN_TIMEOUT_KEEP_ALIVE = int(os.environ.get("UVICORN_TIMEOUT_KEEP_ALIVE", "30"))
    UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN = int(os.environ.get("UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", "15"))
    UVICORN_LIMIT_CONCURRENCY = int(os.environ.get("UVICORN_LIMIT_CONCURRENCY", "100"))
    UVICORN_WORKERS = int(os.environ.get("UVICORN_WORKERS", "2"))
    UVICORN_TIMEOUT = int(os.environ.get("UVICORN_TIMEOUT", "30"))  # Server-side request timeout
    
    _log("http_server_ready", workers=UVICORN_WORKERS, port=8000, 
         timeout_keep_alive=UVICORN_TIMEOUT_KEEP_ALIVE, limit_concurrency=UVICORN_LIMIT_CONCURRENCY,
         onnx_enabled=inferencer.onnx_session is not None, model_backend=getattr(inferencer, 'model_backend', 'unknown'))
    
    try:
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8000,
            workers=UVICORN_WORKERS,
            timeout_keep_alive=UVICORN_TIMEOUT_KEEP_ALIVE,
            timeout_graceful_shutdown=UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN,
            limit_concurrency=UVICORN_LIMIT_CONCURRENCY,
            log_level="info",
        )
    except KeyboardInterrupt:
        _log("http_server_stopped", reason="user_interrupt")
    except Exception as e:
        _log("http_server_fatal_error", error=str(e))
        raise
    finally:
        # Signal auto-reload thread to stop (no global needed - already module-level)
        globals()['_reload_shutdown'] = True
        if _reload_thread and _reload_thread.is_alive():
            _reload_thread.join(timeout=5)
            _log("auto_reload_thread_joined")
