# predict_container/main.py
from client_utils import get_file, post_file
from kafka_utils import create_producer, create_consumer, produce_message, consume_messages, publish_error
from inferencer import Inferencer
import os
import pickle
import torch
import queue
import threading
import time
import traceback
import pyarrow.parquet as pq
import json as _json
import re

# --- Enrichment Helper for any loaded model (pointer or promotion) ---
def _enrich_loaded_model(service: Inferencer, run_id: str, model_type_hint: str | None = None):
    """Fetch MLflow run params to populate model metadata and sequence lengths.

    Emits a structured log event 'promotion_model_enriched'. Safe no-op on failure.
    """
    try:
        import mlflow, pandas as _pd
        run = mlflow.get_run(run_id)
        params = run.data.params or {}
        # Sequence lengths
        if 'input_sequence_length' in params:
            try: service.input_seq_len = int(params['input_sequence_length'])
            except Exception: pass
        if 'output_sequence_length' in params:
            try: service.output_seq_len = int(params['output_sequence_length'])
            except Exception: pass
        # Model type from params if not provided / missing
        if not model_type_hint:
            model_type_hint = params.get('model_type') or params.get('model')
        if model_type_hint and not service.model_type:
            service.model_type = model_type_hint.upper()
        # Reuse existing detection logic from Inferencer
        try:
            row_data = {f"params.{k}": v for k, v in params.items()}
            series = _pd.Series(row_data)
            detected_type, detected_class = service._detect_model_type(series)  # type: ignore[attr-defined]
            if detected_type and not service.model_type:
                service.model_type = detected_type
            if detected_class:
                service.model_class = detected_class
        except Exception as det_err:  # noqa: BLE001
            print({"service": "inference", "event": "model_type_detection_fail", "error": str(det_err)})
        # Hard default for common deep models
        if not service.model_class and service.model_type in {"GRU", "LSTM", "TCN", "TETS"}:
            service.model_class = "pytorch"
        print({
            "service": "inference",
            "event": "promotion_model_enriched",
            "run_id": run_id,
            "model_type": service.model_type,
            "model_class": service.model_class,
            "input_seq_len": service.input_seq_len,
            "output_seq_len": service.output_seq_len
        })
    except Exception as enrich_err:  # noqa: BLE001
        print({"service": "inference", "event": "promotion_model_enrich_fail", "run_id": run_id, "error": str(enrich_err)})

# --- Helper: Robust JSON extraction (supports multipart/form-data wrappers) ---
def _extract_json_from_raw(raw: bytes | str):  # returns tuple(success_bool, payload_dict_or_none, debug_message)
    try:
        if isinstance(raw, (bytes, bytearray)):
            text = raw.decode('utf-8', errors='ignore')
        else:
            text = raw
        # First fast path: direct JSON
        try:
            return True, _json.loads(text), 'direct'
        except Exception:
            pass
        # Detect multipart boundary markers
        if 'Content-Disposition:' in text and '--' in text.splitlines()[0]:
            # Heuristic: extract the largest JSON bracket section
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end+1]
                try:
                    parsed = _json.loads(candidate)
                    return True, parsed, 'multipart_brace_extract'
                except Exception as je:  # noqa: BLE001
                    return False, None, f'brace_extract_failed: {je}'
            # Fallback: regex for JSON object blocks
            obj_matches = re.findall(r'\{[^{}]*model_uri[^{}]*\}', text)
            for m in obj_matches:
                try:
                    parsed = _json.loads(m)
                    if isinstance(parsed, dict) and 'model_uri' in parsed:
                        return True, parsed, 'multipart_regex_extract'
                except Exception:
                    continue
            return False, None, 'multipart_detected_no_json'
        # Fallback generic brace search even if not multipart (maybe trailing headers)
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return True, _json.loads(candidate), 'brace_fallback'
            except Exception as fe:  # noqa: BLE001
                return False, None, f'brace_fallback_failed: {fe}'
        return False, None, 'no_json_detected'
    except Exception as outer:  # noqa: BLE001
        return False, None, f'unhandled_extract_error:{outer}'

# --- Environment Variables ---
GATEWAY_URL = os.environ.get("GATEWAY_URL")
if not GATEWAY_URL:
    raise TypeError("Environment variable, GATEWAY_URL, not defined")
PREPROCESSING_TOPIC = os.environ.get("CONSUMER_TOPIC_0") # Topic for preprocessed data claim checks
if not PREPROCESSING_TOPIC:
    raise TypeError("Environment variable, PREPROCESSING_TOPIC, not defined")
TRAINING_TOPIC = os.environ.get("CONSUMER_TOPIC_1") # Topic for trained model claim checks
PROMOTION_TOPIC = os.environ.get("PROMOTION_TOPIC", "model-selected")
if not TRAINING_TOPIC:
    raise TypeError("Environment variable, TRAINING_TOPIC, not defined")
CONSUMER_GROUP_ID = os.environ.get("CONSUMER_GROUP_ID", "inference_group") # Consumer Group ID
if not CONSUMER_GROUP_ID:
    raise TypeError("Environment variable, CONSUMER_GROUP_ID, not defined")
PRODUCER_TOPIC = os.environ.get("PRODUCER_TOPIC") # Topic for inference results
if not PRODUCER_TOPIC:
    raise TypeError("Environment variable, PRODUCER_TOPIC, not defined")

TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure required buckets exist (idempotent) so inference logging & MLflow artifacts don't fail downstream.
def _ensure_buckets():
    try:
        import boto3, json
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        required = [
            os.environ.get("MLFLOW_ARTIFACT_BUCKET", "mlflow"),
            os.environ.get("PROMOTION_BUCKET", "model-promotion"),
            # New default bucket for structured JSONL inference logs
            os.environ.get("INFERENCE_LOG_BUCKET", "inference-logs"),
        ]
        existing = {b.get('Name') for b in s3.list_buckets().get('Buckets', [])}
        for b in required:
            if b not in existing:
                try:
                    s3.create_bucket(Bucket=b)
                    print(json.dumps({"service": "inference", "event": "bucket_created", "bucket": b}), flush=True)
                except Exception as ce:  # noqa: BLE001
                    print(json.dumps({"service": "inference", "event": "bucket_create_fail", "bucket": b, "error": str(ce)}), flush=True)
            else:
                print(json.dumps({"service": "inference", "event": "bucket_exists", "bucket": b}), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"Bucket ensure error (inference): {e}")

_ensure_buckets()

# --- Kafka Message Queue ---
message_queue = queue.Queue() # A single queue to hold messages from both consumers

# --- Kafka Producer for Inference Output and DLQ ---
producer = create_producer()
dlq_topic = f"DLQ-{PRODUCER_TOPIC}"

# --- Kafka Callback Functions Factory ---
def _kafka_callback_factory(service_instance: Inferencer, source_name: str, message_queue_ref: queue.Queue):
    """Creates a callback function for Kafka consumers to put messages into the shared queue."""
    def callback(message):
        print(f"\nConsumer received {source_name} message with key: {message.key} and added to queue.")
        message_queue_ref.put({"source": source_name, "message": message})
    return callback

# --- Worker Thread Function ---
def message_handler(service: Inferencer, message_queue: queue.Queue):
    """
    Worker thread function that processes messages from the queue.
    It dispatches tasks based on the message source (training or preprocessing).
    """
    print("Inference worker thread started. Waiting for messages in queue...")
    while True:
        try:
            queue_item = message_queue.get()
            source = queue_item.get("source")
            message = queue_item.get("message")

            print(f"Inference worker received message from {source} queue with key: {message.key}")

            if source == "training":
                claim_check = message.value
                operation = claim_check.get("operation")
                status = claim_check.get("status")
                experiment = claim_check.get("experiment")
                run_name = claim_check.get("run_name")

                run_inference_on_train = os.environ.get("RUN_INFERENCE_ON_TRAIN_SUCCESS", "1").lower() in {"1","true","yes"}
                allowed = {m.strip().upper() for m in os.environ.get("ALLOWED_INFERENCE_MODELS", "").split(',') if m.strip()}
                if operation and (status == "SUCCESS") and experiment and run_name:
                    print(f"Inference worker attempting to load new model for experiment '{experiment}', run '{run_name}'.")
                    # Capture config hash for enriched logging downstream
                    try:
                        service.current_config_hash = claim_check.get("config_hash")
                    except Exception:
                        service.current_config_hash = None
                    service.load_model(experiment, run_name)
                    # Only run inference if gating allows and model type is allowed (if list provided)
                    if run_inference_on_train:
                        model_type_upper = (service.model_type or service.current_run_name or "").upper()
                        if (not allowed) or (model_type_upper in allowed):
                            if service.df is not None:
                                service.perform_inference(service.df)
                        else:
                            print({
                                "service": "inference",
                                "event": "skip_model_type_not_allowed",
                                "model_type": model_type_upper,
                                "allowed": list(allowed)
                            })
                else:
                    print(f"Inference worker WARN: Training message received without complete details or success status: {claim_check}")
                    publish_error(
                        service.producer,
                        service.dlq_topic,
                        "Training Message Parse",
                        "Failure",
                        "Incomplete training claim check",
                        claim_check
                    )
            elif source == "preprocessing":
                claim_check = message.value
                # Support both legacy and new claim shapes
                bucket = claim_check.get("bucket")
                object_key = claim_check.get("object_key") or claim_check.get("object")
                operation = claim_check.get("operation")  # Optional

                if bucket and object_key:
                    print(f"Inference worker fetching data from object store: s3://{bucket}/{object_key}")
                    try:
                        parquet_bytes = get_file(service.gateway_url, bucket, object_key)
                        table = pq.read_table(source=parquet_bytes)
                        service.df = table.to_pandas()

                        if service.current_model is not None:
                            service.perform_inference(service.df)
                        else:
                            print("Model not yet loaded; stored dataframe for later inference.")
                    except Exception as e:
                        print(f"Inference worker error fetching, parsing, or during inference for {object_key}: {e}")
                        traceback.print_exc()
                        publish_error(
                            service.producer,
                            service.dlq_topic,
                            "Data Fetch/Inference",
                            "Failure",
                            str(e),
                            {"bucket": bucket, "object_key": object_key}
                        )
                else:
                    print(f"Inference worker WARN: Preprocessing message missing bucket/object fields: {claim_check}")
                    publish_error(
                        service.producer,
                        service.dlq_topic,
                        "Preprocessing Message Parse",
                        "Failure",
                        "Incomplete preprocessing claim check",
                        claim_check
                    )
            elif source == "promotion":
                claim_check = message.value
                run_id = claim_check.get("run_id")
                model_uri = claim_check.get("model_uri")
                model_type = claim_check.get("model_type")
                print(f"Promotion message received for run_id={run_id}, model_type={model_type}")
                try:
                    if model_uri and run_id:
                        from mlflow import pyfunc
                        attempted = []
                        # Try original URI then fallback 'model' subpath if not already that
                        uri_candidates = [model_uri]
                        if "/model" not in model_uri.split('/')[-1]:
                            # Append fallback only if not already pointing to 'model'
                            base = model_uri.rsplit('/', 1)[0]
                            uri_candidates.append(f"{base}/model")
                        loaded = False
                        for cand in uri_candidates:
                            try:
                                print(f"Loading promoted model via URI: {cand}")
                                promoted_model = pyfunc.load_model(cand)
                                service.current_model = promoted_model
                                # Update core identity metadata BEFORE enrichment so logs reflect correct model_type
                                service.current_run_id = run_id  # critical: previously missing caused stale run_id usage
                                # Capture config hash if provided
                                try:
                                    service.current_config_hash = claim_check.get("config_hash")
                                except Exception:
                                    service.current_config_hash = getattr(service, "current_config_hash", None)
                                # Normalize model type (uppercase for consistency)
                                normalized_type = (model_type or '').upper()
                                if normalized_type:
                                    service.model_type = normalized_type
                                # Mirror run name to model_type for downstream expectations
                                service.current_run_name = service.model_type or normalized_type or ''
                                service.current_experiment_name = claim_check.get("experiment", "Default")
                                print(f"✅ Promoted model loaded from {cand}")
                                loaded = True
                                break
                            except Exception as le:  # noqa: BLE001
                                print(f"Promotion model load attempt failed for {cand}: {le}")
                                attempted.append({"candidate": cand, "error": str(le)})
                                continue
                        if not loaded:
                            raise Exception(f"All promotion model load attempts failed: {attempted}")
                        # Enrich model metadata & seq lens
                        if run_id:
                            # Pass currently set service.model_type so enrichment does not re-use stale value
                            _enrich_loaded_model(service, run_id, service.model_type or model_type)
                        if service.current_model is not None and service.df is not None:
                            service.perform_inference(service.df)
                    else:
                        print("Promotion message missing run_id/model_uri; sending to DLQ")
                        publish_error(service.producer, service.dlq_topic, "Promotion Message Parse", "Failure", "Incomplete promotion message", claim_check)
                except Exception as e:
                    publish_error(service.producer, service.dlq_topic, "Promotion Load", "Failure", str(e), claim_check)
            else:
                print(f"Inference worker WARN: Unknown message source: {source}. Message: {message.value}")
                publish_error(
                    service.producer,
                    service.dlq_topic,
                    "Unknown Message Source",
                    "Failure",
                    f"Message from unknown source '{source}'",
                    message.value
                )

        except Exception as e:
            print(f"Inference worker failed to process message from queue: {e}")
            traceback.print_exc()
            publish_error(
                service.producer,
                service.dlq_topic,
                "Queue Processing",
                "Failure",
                str(e),
                "No specific payload (queue error)"
            )
        finally:
            message_queue.task_done()


inferencer = Inferencer(GATEWAY_URL, producer, dlq_topic, PRODUCER_TOPIC)

# Flag to ensure we only start background threads once (important when module is imported by API server)
_RUNTIME_STARTED = False

# --- Optional Startup Preload & Promotion Autoload ---
def _preload_test_dataframe(service: Inferencer):
    """Attempt to preload the test parquet so that if a model is recovered (promotion) we can infer immediately.
    Safe best-effort; logs JSON event either way.
    """
    try:
        enable = os.environ.get("INFERENCE_PRELOAD_TEST", "1").lower() in {"1", "true", "yes"}
        if not enable:
            return
        test_bucket = os.environ.get("PROCESSED_BUCKET", "processed-data")
        test_key = os.environ.get("TEST_OBJECT_KEY", "test_processed_data.parquet")
        parquet_bytes = get_file(service.gateway_url, test_bucket, test_key)
        table = pq.read_table(source=parquet_bytes)
        service.df = table.to_pandas()
        print({"service": "inference", "event": "preload_test_success", "rows": int(len(service.df.index)), "cols": int(len(service.df.columns)), "object_key": test_key})
    except Exception as e:  # noqa: BLE001
        print({"service": "inference", "event": "preload_test_fail", "error": str(e)})

def _attempt_load_promoted(service: Inferencer):
    """Try to load the last promoted model (current.json) so restarts don't require replaying Kafka history.
    Will look under identifier-specific path then global fallback.
    """
    try:
        enable = os.environ.get("INFERENCE_AUTOLOAD_PROMOTED", "1").lower() in {"1", "true", "yes"}
        if not enable:
            return
        import json
        from mlflow import pyfunc
        promotion_bucket = os.environ.get("PROMOTION_BUCKET", "model-promotion")
        identifier = os.environ.get("IDENTIFIER", "") or "global"
        # New canonical root-level pointer first, then legacy global/<...> and identifier-specific
        candidate_keys = ["current.json"]
        # Preserve legacy order after root-level for backward compatibility / dashboards
        if "global/current.json" not in candidate_keys:
            candidate_keys.append("global/current.json")
        scoped = f"{identifier}/current.json"
        if identifier != "global" and scoped not in candidate_keys:
            candidate_keys.append(scoped)
        loaded_payload = None
        for key in candidate_keys:
            try:
                obj = get_file(service.gateway_url, promotion_bucket, key)
                raw = obj.getvalue() if hasattr(obj, 'getvalue') else obj
                ok, payload, mode = _extract_json_from_raw(raw)
                if ok and isinstance(payload, dict):
                    loaded_payload = payload
                    print({"service": "inference", "event": "promotion_manifest_found", "object_key": key, "config_hash": payload.get("config_hash"), "parse_mode": mode})
                    print({"service": "inference", "event": "promotion_pointer_parsed", "run_id": payload.get('run_id'), "model_type": payload.get('model_type'), "config_hash": payload.get('config_hash')})
                    break
                else:
                    print({"service": "inference", "event": "promotion_pointer_parse_fail", "object_key": key, "reason": mode})
            except Exception as e:  # noqa: BLE001
                print({"service": "inference", "event": "promotion_pointer_fetch_fail", "object_key": key, "error": str(e)})
                continue
        if not loaded_payload:
            # Keep legacy event name for existing dashboards; also emit new event alias
            print({"service": "inference", "event": "promotion_manifest_missing", "candidates": candidate_keys})
            print({"service": "inference", "event": "promotion_manifest_absent", "candidates": candidate_keys})
            return
        model_uri = loaded_payload.get("model_uri")
        run_id = loaded_payload.get("run_id")
        model_type = loaded_payload.get("model_type")
        experiment = loaded_payload.get("experiment", "Default")
        cfg_hash = loaded_payload.get("config_hash")
        if not model_uri or not run_id:
            print({"service": "inference", "event": "promotion_manifest_incomplete", "payload_keys": list(loaded_payload.keys())})
            return
        # Attempt direct load; if fails and not already endswith /model, append /model as fallback.
        uri_candidates = [model_uri]
        if not model_uri.rstrip('/').endswith('/model'):
            uri_candidates.append(model_uri.rstrip('/') + '/model')
        for cand in uri_candidates:
            try:
                print(f"Loading promoted model at startup via URI: {cand}")
                service.current_model = pyfunc.load_model(cand)
                service.current_run_name = model_type or ''
                service.current_experiment_name = experiment
                service.current_config_hash = cfg_hash
                service.model_type = model_type or ''
                service.current_run_id = run_id
                print({"service": "inference", "event": "promotion_model_loaded_startup", "model_uri": cand, "run_id": run_id, "config_hash": cfg_hash})
                if run_id:
                    _enrich_loaded_model(service, run_id, model_type)
                break
            except Exception as le:  # noqa: BLE001
                print({"service": "inference", "event": "promotion_model_load_fail_startup", "candidate": cand, "error": str(le)})
        # If both model & data present now, run inference immediately.
        if service.current_model is not None and service.df is not None:
            try:
                service.perform_inference(service.df)
            except Exception as ie:  # noqa: BLE001
                print({"service": "inference", "event": "startup_inference_fail", "error": str(ie)})
    except Exception as e:  # noqa: BLE001
        print({"service": "inference", "event": "promotion_autoload_fail", "error": str(e)})

_preload_test_dataframe(inferencer)

def _load_promoted_pointer(service: Inferencer):
    """Load the most recent promoted model based on a pointer manifest.

    Primary lookup path (new requirement):
      - Bucket: PROMOTION_BUCKET (default model-promotion), key: current.json

    Backward compatibility fallbacks (legacy structure still supported):
      - global/current.json
      - <identifier>/current.json (if IDENTIFIER provided and not empty)
    """
    import json
    promotion_bucket = os.environ.get("PROMOTION_BUCKET", "model-promotion")
    identifier = os.environ.get("IDENTIFIER", "")
    # Ordered candidate keys: new root pointer first, then legacy variants
    candidates = ["current.json"]  # root-level canonical
    # Legacy global path
    if "global/current.json" not in candidates:
        candidates.append("global/current.json")
    if identifier and f"{identifier}/current.json" not in candidates:
        candidates.append(f"{identifier}/current.json")

    loaded_payload = None
    for key in candidates:
        try:
            obj = get_file(service.gateway_url, promotion_bucket, key)  # type: ignore[name-defined]
            raw = obj.getvalue() if hasattr(obj, 'getvalue') else obj
            ok, payload, mode = _extract_json_from_raw(raw)
            if ok and isinstance(payload, dict):
                loaded_payload = payload
                print({"service": "inference", "event": "promotion_manifest_loaded", "object_key": key, "config_hash": payload.get("config_hash"), "parse_mode": mode})
                print({"service": "inference", "event": "promotion_pointer_parsed", "run_id": payload.get('run_id'), "model_type": payload.get('model_type'), "config_hash": payload.get('config_hash')})
                break
            else:
                print({"service": "inference", "event": "promotion_pointer_parse_fail", "object_key": key, "reason": mode})
        except Exception as e:  # noqa: BLE001
            print({"service": "inference", "event": "promotion_pointer_fetch_fail", "object_key": key, "error": str(e)})
            continue

    if not loaded_payload:
        print({"service": "inference", "event": "promotion_manifest_absent", "message": "No promotion manifest found — waiting for next promotion event."})
        return

    model_uri = loaded_payload.get("model_uri")
    run_id = loaded_payload.get("run_id")
    model_type = loaded_payload.get("model_type")
    experiment = loaded_payload.get("experiment", "Default")
    cfg_hash = loaded_payload.get("config_hash")
    if not model_uri or not run_id:
        print({"service": "inference", "event": "promotion_manifest_incomplete", "payload_keys": list(loaded_payload.keys())})
        return

    from mlflow import pyfunc  # lazy import inside function
    uri_candidates = [model_uri]
    if not model_uri.rstrip('/').endswith('/model'):
        uri_candidates.append(model_uri.rstrip('/') + '/model')
    for cand in uri_candidates:
        try:
            print(f"Loading promoted model at startup via URI: {cand}")
            service.current_model = pyfunc.load_model(cand)
            service.current_run_name = model_type or ''
            service.model_type = model_type or ''
            service.current_experiment_name = experiment
            service.current_config_hash = cfg_hash
            service.current_run_id = run_id
            print({"service": "inference", "event": "promotion_model_loaded_startup", "model_uri": cand, "run_id": run_id, "config_hash": cfg_hash})
            if run_id:
                _enrich_loaded_model(service, run_id, model_type)
            break
        except Exception as le:  # noqa: BLE001
            print({"service": "inference", "event": "promotion_model_load_fail_startup", "candidate": cand, "error": str(le)})
    if os.getenv("DISABLE_STARTUP_INFERENCE", "0") == "1":
        print({"service": "inference", "event": "startup_inference_skipped"})
    elif service.current_model is not None and service.df is not None:
        try:
            service.perform_inference(service.df)
        except Exception as ie:  # noqa: BLE001
            print({"service": "inference", "event": "startup_inference_fail", "error": str(ie)})

def _start_runtime():
    """Start background worker & Kafka consumers.

    Safe to call multiple times; subsequent calls are no-ops.
    We intentionally avoid an infinite sleep loop so that importing this
    module from the FastAPI app does not block the request thread.
    """
    global _RUNTIME_STARTED, training_consumer, preprocessing_consumer, promotion_consumer
    if _RUNTIME_STARTED:
        return
    _RUNTIME_STARTED = True
    print({"service": "inference", "event": "runtime_start", "note": "Starting background threads"})
    _load_promoted_pointer(inferencer)

    # Start the worker thread, passing the service instance and queue
    worker_thread = threading.Thread(
        target=message_handler,
        args=(inferencer, message_queue),
        daemon=True
    )
    worker_thread.start()

    # Create and start consumers in their own threads (training, preprocessing, promotion)
    training_consumer = create_consumer(TRAINING_TOPIC, CONSUMER_GROUP_ID)
    training_callback_func = _kafka_callback_factory(inferencer, "training", message_queue)
    training_consumer_thread = threading.Thread(target=consume_messages, args=(training_consumer, training_callback_func), daemon=True)
    training_consumer_thread.start()
    print(f"Started Kafka consumer for training topic: {TRAINING_TOPIC}")

    preprocessing_consumer = create_consumer(PREPROCESSING_TOPIC, CONSUMER_GROUP_ID)
    preprocessing_callback_func = _kafka_callback_factory(inferencer, "preprocessing", message_queue)
    preprocessing_consumer_thread = threading.Thread(target=consume_messages, args=(preprocessing_consumer, preprocessing_callback_func), daemon=True)
    preprocessing_consumer_thread.start()
    print(f"Started Kafka consumer for preprocessing topic: {PREPROCESSING_TOPIC}")

    promotion_consumer = create_consumer(PROMOTION_TOPIC, CONSUMER_GROUP_ID)
    promotion_callback_func = _kafka_callback_factory(inferencer, "promotion", message_queue)
    promotion_consumer_thread = threading.Thread(target=consume_messages, args=(promotion_consumer, promotion_callback_func), daemon=True)
    promotion_consumer_thread.start()
    print(f"Started Kafka consumer for promotion topic: {PROMOTION_TOPIC}")


def _graceful_shutdown():
    try:
        if 'training_consumer' in globals() and training_consumer:
            training_consumer.close()
        if 'preprocessing_consumer' in globals() and preprocessing_consumer:
            preprocessing_consumer.close()
        if 'promotion_consumer' in globals() and promotion_consumer:
            promotion_consumer.close()
        if producer:
            producer.close()
    finally:
        print("Kafka consumers and producer closed.")


# Start runtime immediately on import unless explicitly disabled (e.g., tests)
if os.environ.get("INFERENCE_AUTOSTART", "1") in {"1", "true", "TRUE"}:
    try:
        _start_runtime()
    except Exception as e:  # noqa: BLE001
        print({"service": "inference", "event": "runtime_start_fail", "error": str(e)})


if __name__ == "__main__":
    # When executed as a script, keep process alive.
    try:
        print({"service": "inference", "event": "main_loop_enter"})
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Inference container stopped by user.")
    finally:
        _graceful_shutdown()

