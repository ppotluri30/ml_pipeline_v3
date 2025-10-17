# predict_container/main.py
from client_utils import get_file, post_file
from data_utils import strip_timezones
from kafka_utils import (
    create_producer,
    create_consumer,
    create_consumer_configurable,
    produce_message,
    consume_messages,
    publish_error,
    commit_offsets_sync,
)
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

if os.environ.get("DISABLE_BUCKET_ENSURE", "0").lower() not in {"1","true","yes"}:
    _ensure_buckets()

# --- Kafka Message Queue ---
# --- Queue configuration (bounded by env when enabled) ---
USE_BOUNDED_QUEUE = os.environ.get("USE_BOUNDED_QUEUE", "false").lower() in {"1", "true", "yes"}
QUEUE_MAXSIZE = int(os.environ.get("QUEUE_MAXSIZE", "512"))
message_queue = queue.Queue(maxsize=QUEUE_MAXSIZE) if USE_BOUNDED_QUEUE else queue.Queue()
# Per-source commit queues: filled by worker after successful processing
commit_queues = {
    "training": queue.Queue(),
    "preprocessing": queue.Queue(),
    "promotion": queue.Queue(),
}

# Track the most recent promoted pointer so runtime decisions can restrict loads
_PROMOTED_STATE = {
    "run_id": None,
    "model_uri": None,
    "config_hash": None,
    "model_type": None,
    "experiment": None,
}

# --- Kafka Producer for Inference Output and DLQ ---
producer = create_producer()
dlq_topic = f"DLQ-{PRODUCER_TOPIC}"

# --- Kafka Callback Functions Factory ---
def _kafka_callback_factory(service_instance: Inferencer, source_name: str, message_queue_ref: queue.Queue):
    """Creates a callback function for Kafka consumers to put messages into the shared queue."""
    def callback(message):
        # Enqueue or block depending on bounded queue setting
        if USE_BOUNDED_QUEUE and message_queue_ref.full():
            # Drop into controlled backpressure handled by pause/resume; still try to enqueue without blocking
            try:
                message_queue_ref.put({
                    "source": source_name,
                    "message": message,
                    "tp": getattr(message, 'topic', None),
                    "partition": getattr(message, 'partition', None),
                    "offset": getattr(message, 'offset', None),
                }, block=False)
            except queue.Full:
                # As a last resort, block briefly to avoid tight spin
                message_queue_ref.put({
                    "source": source_name,
                    "message": message,
                    "tp": getattr(message, 'topic', None),
                    "partition": getattr(message, 'partition', None),
                    "offset": getattr(message, 'offset', None),
                }, timeout=0.5)
        else:
            message_queue_ref.put({
                "source": source_name,
                "message": message,
                "tp": getattr(message, 'topic', None),
                "partition": getattr(message, 'partition', None),
                "offset": getattr(message, 'offset', None),
            })
        print({
            "service": "inference",
            "event": "queue_enqueued",
            "source": source_name,
            "depth": message_queue_ref.qsize(),
            "bounded": int(USE_BOUNDED_QUEUE),
            "maxsize": QUEUE_MAXSIZE if USE_BOUNDED_QUEUE else -1
        })
    return callback

# --- Worker Thread Function ---
def message_handler(service: Inferencer, message_queue: queue.Queue):
    """
    Worker thread function that processes messages from the queue.
    It dispatches tasks based on the message source (training or preprocessing).
    """
    print("Inference worker thread started. Waiting for messages in queue...")
    ENABLE_TTL = os.environ.get("ENABLE_TTL", "false").lower() in {"1","true","yes"}
    ENABLE_MICROBATCH = os.environ.get("ENABLE_MICROBATCH", "false").lower() in {"1","true","yes"}
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
    BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", "25"))
    while True:
        try:
            if ENABLE_MICROBATCH:
                # Drain up to BATCH_SIZE or until timeout
                batch = []
                first = message_queue.get()
                batch.append(first)
                start_t = time.time()
                while len(batch) < BATCH_SIZE:
                    timeout_sec = max(0.0, BATCH_TIMEOUT_MS/1000.0 - (time.time() - start_t))
                    if timeout_sec <= 0:
                        break
                    try:
                        item = message_queue.get(timeout=timeout_sec)
                        batch.append(item)
                    except queue.Empty:
                        break
                items = batch
            else:
                items = [message_queue.get()]
            # Track highest processed offsets per source/topic-partition
            processed_offsets = {"training": {}, "preprocessing": {}, "promotion": {}}
            for queue_item in items:
                source = queue_item.get("source")
                message = queue_item.get("message")
                tp_name = queue_item.get("tp")
                partition = queue_item.get("partition")
                offset = queue_item.get("offset")

                print(f"Inference worker received message from {source} queue with key: {message.key}")

                # TTL handling via Kafka headers
                if ENABLE_TTL:
                    try:
                        headers_list = message.headers or []  # type: ignore[attr-defined]
                        headers = { (k.decode() if isinstance(k, (bytes, bytearray)) else k): v for k, v in headers_list }
                        if 'deadline_ms' in headers:
                            try:
                                raw_val = headers['deadline_ms']
                                deadline_ms = int(raw_val.decode()) if isinstance(raw_val, (bytes, bytearray)) else int(raw_val)
                                if int(time.time() * 1000) > deadline_ms:
                                    print({"service": "inference", "event": "ttl_expired", "source": source, "key": message.key})
                                    # Skip processing; finalizer will mark task_done and commit loop will proceed
                                    continue
                            except Exception as _:
                                pass
                    except Exception:
                        pass

                if source == "training":
                    claim_check = message.value
                    operation = claim_check.get("operation")
                    status = claim_check.get("status")
                    run_id = claim_check.get("run_id")
                    experiment = claim_check.get("experiment")
                    run_name = claim_check.get("run_name")

                    if not (operation and status == "SUCCESS" and run_id):
                        print({
                            "service": "inference",
                            "event": "training_claim_incomplete",
                            "payload_keys": list(claim_check.keys()) if isinstance(claim_check, dict) else None
                        })
                        publish_error(
                            service.producer,
                            service.dlq_topic,
                            "Training Message Parse",
                            "Failure",
                            "Incomplete training claim check",
                            claim_check
                        )
                        continue

                    promoted_run_id = _PROMOTED_STATE.get("run_id")
                    if promoted_run_id is None:
                        print({
                            "service": "inference",
                            "event": "training_claim_refresh_promoted",
                            "note": "Promotion state missing; attempting pointer reload"
                        })
                        _load_promoted_pointer(service)
                        promoted_run_id = _PROMOTED_STATE.get("run_id")

                    if promoted_run_id != run_id:
                        print({
                            "service": "inference",
                            "event": "training_claim_skipped",
                            "reason": "run_id_not_promoted",
                            "run_id": run_id,
                            "promoted_run_id": promoted_run_id
                        })
                        continue

                    if service.current_model is None or getattr(service, "current_run_id", None) != run_id:
                        print({
                            "service": "inference",
                            "event": "training_claim_promoted_reload",
                            "run_id": run_id,
                            "experiment": experiment,
                            "run_name": run_name
                        })
                        _load_promoted_pointer(service)
                        if getattr(service, "current_run_id", None) != run_id:
                            print({
                                "service": "inference",
                                "event": "training_claim_reload_mismatch",
                                "expected_run_id": run_id,
                                "current_run_id": getattr(service, "current_run_id", None)
                            })
                    else:
                        print({
                            "service": "inference",
                            "event": "training_claim_already_promoted",
                            "run_id": run_id
                        })

                    run_inference_on_train = os.environ.get("RUN_INFERENCE_ON_TRAIN_SUCCESS", "1").lower() in {"1","true","yes"}
                    allowed = {m.strip().upper() for m in os.environ.get("ALLOWED_INFERENCE_MODELS", "").split(',') if m.strip()}
                    if run_inference_on_train and service.current_model is not None:
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
                            service.df, tz_meta = strip_timezones(service.df)
                            if tz_meta.get("index") or tz_meta.get("columns"):
                                print({
                                    "service": "inference",
                                    "event": "preprocess_dataframe_tz_normalized",
                                    "object_key": object_key,
                                    "index_adjusted": bool(tz_meta.get("index")),
                                    "columns_adjusted": tz_meta.get("columns", []),
                                })

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
                                _PROMOTED_STATE.update({
                                    "run_id": run_id,
                                    "model_uri": cand,
                                    "config_hash": claim_check.get("config_hash"),
                                    "model_type": service.model_type,
                                    "experiment": service.current_experiment_name,
                                })
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

                # record processed offset
                try:
                    if tp_name is not None and partition is not None and offset is not None:
                        key = (str(tp_name), int(partition))
                        current = processed_offsets[source].get(key, -1)
                        if int(offset) > current:
                            processed_offsets[source][key] = int(offset)
                except Exception:
                    pass

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
            # Mark all drained items as done
            try:
                if ENABLE_MICROBATCH:
                    for _ in items:
                        message_queue.task_done()
                else:
                    message_queue.task_done()
            except Exception:
                pass
            # After marking tasks done, publish processed offsets for commit
            try:
                for src, tpmap in processed_offsets.items():
                    if tpmap:
                        commit_queues[src].put(tpmap)
            except Exception:
                pass


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

# NOTE: Previously this preload ran at import time which could block callers
# that import this module (for example the FastAPI app via `from main import inferencer`).
# Running long IO during import prevents the API from responding to readiness
# probes. We intentionally DO NOT call _preload_test_dataframe here; the preload
# is invoked from _start_runtime() when the runtime is started explicitly.


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

    _PROMOTED_STATE.update({
        "run_id": run_id,
        "model_uri": model_uri,
        "config_hash": cfg_hash,
        "model_type": model_type,
        "experiment": experiment,
    })
    print({"service": "inference", "event": "promotion_state_cached", "run_id": run_id, "model_uri": model_uri})

    # Avoid redundant reloads if we already have the promoted run active
    if service.current_model is not None and getattr(service, "current_run_id", None) == run_id:
        service.current_experiment_name = experiment
        service.model_type = model_type or service.model_type
        service.current_config_hash = cfg_hash or service.current_config_hash
        print({
            "service": "inference",
            "event": "promotion_model_already_active",
            "run_id": run_id
        })
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
    # Run the preload here (was previously at import time). Running it during
    # runtime start ensures the API can be imported quickly and /ready can
    # respond without being blocked by IO.
    try:
        _preload_test_dataframe(inferencer)
    except Exception as e:
        print({"service": "inference", "event": "preload_run_error", "error": str(e)})

    _load_promoted_pointer(inferencer)

    # Start the worker thread, passing the service instance and queue
    worker_thread = threading.Thread(
        target=message_handler,
        args=(inferencer, message_queue),
        daemon=True
    )
    worker_thread.start()

    # Consumer configuration flags
    USE_MANUAL_COMMIT = os.environ.get("USE_MANUAL_COMMIT", "false").lower() in {"1","true","yes"}
    FETCH_MAX_WAIT_MS = int(os.environ.get("FETCH_MAX_WAIT_MS", "50"))
    MAX_POLL_RECORDS = int(os.environ.get("MAX_POLL_RECORDS", "64"))

    # Backpressure thresholds (percent of QUEUE_MAXSIZE)
    PAUSE_THRESHOLD_PCT = float(os.environ.get("PAUSE_THRESHOLD_PCT", "80"))
    RESUME_THRESHOLD_PCT = float(os.environ.get("RESUME_THRESHOLD_PCT", "50"))

    def _consumer_loop(topic_name: str, source_name: str):
        consumer = create_consumer_configurable(topic_name, CONSUMER_GROUP_ID, enable_auto_commit=not USE_MANUAL_COMMIT)
        assigned = None
        paused = False
        try:
            print({"service": "inference", "event": "consumer_loop_start", "topic": topic_name, "manual_commit": int(USE_MANUAL_COMMIT)})
            while True:
                # Pause/resume by queue depth (only if bounded)
                if USE_BOUNDED_QUEUE:
                    depth = message_queue.qsize()
                    cap = QUEUE_MAXSIZE if USE_BOUNDED_QUEUE else 0
                    if cap:
                        pct = 100.0 * depth / cap
                        if pct >= PAUSE_THRESHOLD_PCT:
                            assigned = assigned or consumer.assignment()
                            if assigned and not paused:
                                consumer.pause(*assigned)
                                print({"service": "inference", "event": "consumer_paused", "topic": topic_name, "depth": depth, "pct": round(pct,2)})
                                paused = True
                        elif pct <= RESUME_THRESHOLD_PCT:
                            assigned = assigned or consumer.assignment()
                            if assigned and paused:
                                consumer.resume(*assigned)
                                print({"service": "inference", "event": "consumer_resumed", "topic": topic_name, "depth": depth, "pct": round(pct,2)})
                                paused = False

                # Poll a batch
                records = consumer.poll(timeout_ms=FETCH_MAX_WAIT_MS, max_records=MAX_POLL_RECORDS)
                if not records:
                    # Even if no new records, apply any pending commits after processing
                    if USE_MANUAL_COMMIT:
                        try:
                            from kafka.structs import TopicPartition, OffsetAndMetadata  # type: ignore
                            drained = 0
                            while True:
                                tpmap = commit_queues[source_name].get_nowait()
                                commit_map = {}
                                for (tp_topic, tp_part), last_offset in tpmap.items():
                                    tp = TopicPartition(tp_topic, tp_part)
                                    # Include leader_epoch (-1) for compatibility with kafka-python versions that
                                    # require an integer leader_epoch; -1 means "no epoch".
                                    commit_map[tp] = OffsetAndMetadata(last_offset + 1, None, -1)
                                commit_offsets_sync(consumer, commit_map)
                                commit_queues[source_name].task_done()
                                drained += 1
                            if drained:
                                print({"service": "inference", "event": "commits_applied", "drained_batches": drained, "source": source_name})
                        except queue.Empty:
                            pass
                    continue
                # Track highest offsets per TP for commit (we commit after processing via commit_queues)
                for tp, messages in records.items():
                    for msg in messages:
                        cb = _kafka_callback_factory(inferencer, source_name, message_queue)
                        try:
                            cb(msg)
                        except Exception as e:
                            print({"service": "inference", "event": "enqueue_fail", "error": str(e)})

                # Drain processed offsets and commit next offsets (at-least-once after processing)
                if USE_MANUAL_COMMIT:
                    try:
                        drained = 0
                        from kafka.structs import TopicPartition, OffsetAndMetadata  # type: ignore
                        while True:
                            tpmap = commit_queues[source_name].get_nowait()
                            commit_map = {}
                            for (tp_topic, tp_part), last_offset in tpmap.items():
                                tp = TopicPartition(tp_topic, tp_part)
                                # Include leader_epoch (-1) for compatibility with kafka-python versions that
                                # require an integer leader_epoch; -1 means "no epoch".
                                commit_map[tp] = OffsetAndMetadata(last_offset + 1, None, -1)
                            commit_offsets_sync(consumer, commit_map)
                            commit_queues[source_name].task_done()
                            drained += 1
                        if drained:
                            print({"service": "inference", "event": "commits_applied", "drained_batches": drained, "source": source_name})
                    except queue.Empty:
                        pass
        except Exception as e:
            print({"service": "inference", "event": "consumer_loop_error", "topic": topic_name, "error": str(e)})
        finally:
            try: consumer.close()
            except Exception: pass

    # Start consumers with the new loop (training, preprocessing, promotion)
    threading.Thread(target=_consumer_loop, args=(TRAINING_TOPIC, "training"), daemon=True).start()
    print(f"Started Kafka consumer for training topic: {TRAINING_TOPIC}")

    threading.Thread(target=_consumer_loop, args=(PREPROCESSING_TOPIC, "preprocessing"), daemon=True).start()
    print(f"Started Kafka consumer for preprocessing topic: {PREPROCESSING_TOPIC}")

    threading.Thread(target=_consumer_loop, args=(PROMOTION_TOPIC, "promotion"), daemon=True).start()
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


# Do NOT start runtime automatically at import time. Runtime is started by
# the FastAPI app during its non-blocking startup handler so the HTTP server
# binds immediately and background model loading does not block Uvicorn.
def start_runtime_safe():
    try:
        _start_runtime()
    except Exception as e:  # noqa: BLE001
        print({"service": "inference", "event": "runtime_start_fail", "error": str(e)})


if __name__ == "__main__":
    # When executed as a script, keep process alive.
    try:
        print({"service": "inference", "event": "main_loop_enter"})
        # Start runtime when running as the main process (local debug)
        start_runtime_safe()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Inference container stopped by user.")
    finally:
        _graceful_shutdown()

