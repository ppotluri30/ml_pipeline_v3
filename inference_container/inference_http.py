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
        test_df = strip_timezones(test_df)
        _log("preload_test_dataframe_success", rows=len(test_df))
    except Exception as e:
        _log("preload_test_dataframe_error", error=str(e))


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
else:
    _log("startup_warning", message="No promoted model loaded - will serve with empty model")

# Start auto-reload background thread
import threading
_reload_thread = threading.Thread(target=_auto_reload_loop, daemon=True, name="model-auto-reload")
_reload_thread.start()
_log("auto_reload_enabled", check_interval_seconds=_reload_check_interval)


if __name__ == "__main__":
    # Import FastAPI app (it will use our global inferencer instance)
    from api_server import app
    
    # Start Uvicorn server
    import uvicorn
    
    UVICORN_TIMEOUT_KEEP_ALIVE = int(os.environ.get("UVICORN_TIMEOUT_KEEP_ALIVE", "60"))
    UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN = int(os.environ.get("UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", "30"))
    UVICORN_LIMIT_CONCURRENCY = int(os.environ.get("UVICORN_LIMIT_CONCURRENCY", "1000"))
    
    _log("http_server_ready", workers=1, port=8000)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
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
