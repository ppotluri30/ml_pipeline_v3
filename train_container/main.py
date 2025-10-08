"""Training consumer entrypoint (clean rebuild without XAI).

Version marker injected for runtime verification. Increment VERSION when patching.
"""

VERSION = "trainer_v20251002_03"

import numpy as np
import pandas as pd
import mlflow  # type: ignore
import mlflow.pytorch  # type: ignore
import pickle
import tempfile
import os
import threading
import queue
import json
import pyarrow.parquet as pq
import torch
import time
from typing import List, Dict, Any, Tuple, Set

from ml_models import LSTM, GRU, TETS, TCN, EncoderLSTM
from train import prepare_data_loaders, train
from data_utils import window_data, subset_scaler
from client_utils import get_file
from kafka_utils import create_consumer, consume_messages, create_producer, produce_message, publish_error

# Emit version marker immediately so container logs prove fresh code deployment
print(json.dumps({"service": "train", "event": "version_start", "version": VERSION}), flush=True)

# Ensure required S3 (MinIO) buckets exist before any MLflow artifact logging.
def _ensure_buckets():
    try:
        import boto3
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
            os.environ.get("INFERENCE_LOG_BUCKET", "inference-txt-logs"),
        ]
        existing = {b.get('Name') for b in s3.list_buckets().get('Buckets', [])}
        for b in required:
            if b not in existing:
                try:
                    s3.create_bucket(Bucket=b)
                    print(json.dumps({"service": "train", "event": "bucket_created", "bucket": b}), flush=True)
                except Exception as ce:  # noqa: BLE001
                    # Ignore race if another container creates it first
                    print(json.dumps({"service": "train", "event": "bucket_create_fail", "bucket": b, "error": str(ce)}), flush=True)
            else:
                print(json.dumps({"service": "train", "event": "bucket_exists", "bucket": b}), flush=True)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"service": "train", "event": "bucket_ensure_error", "error": str(e)}), flush=True)

_ensure_buckets()


def env_var(var: str, default: Any = None) -> str:
    temp = os.environ.get(var, default)
    if temp is None:
        raise TypeError(f"Environment variable, {var}, not defined")
    return temp


def _jlog(event: str, **extra):
    base = {"service": "train", "event": event}
    for k, v in extra.items():
        if v is not None:
            base[k] = v
    print(json.dumps(base))


def callback(message):
    _jlog("kafka_receive", key=message.key, partition=getattr(message, 'partition', None), offset=getattr(message, 'offset', None))
    message_queue.put(message)


FAILURE_MAX_RETRIES = int(os.environ.get("FAILURE_MAX_RETRIES", "3"))
_failure_counts: Dict[str, int] = {}
_last_run_context: Dict[str, Any] = {}
# Duplicate / loop suppression controls
SKIP_DUPLICATE_CONFIGS = os.environ.get("SKIP_DUPLICATE_CONFIGS", "1").lower() in {"1", "true", "yes"}
# Tracks (MODEL_TYPE, config_hash) already successfully trained this process lifetime
_processed_config_models: Set[Tuple[str, str]] = set()
MAX_PROCESSED_CACHE = int(os.environ.get("DUP_CACHE_MAX", "500"))


def _extract_meta(schema):
    out = {}
    if not schema or not schema.metadata:
        return out
    md = schema.metadata
    if b'preprocess_config' in md:
        try:
            out['preprocess_config'] = json.loads(md[b'preprocess_config'].decode())
        except Exception:  # noqa: BLE001
            pass
    if b'config_hash' in md:
        out['config_hash'] = md[b'config_hash'].decode()
    if b'scaler_object' in md:
        try:
            out['scaler_object'] = pickle.loads(md[b'scaler_object'])
        except Exception:  # noqa: BLE001
            pass
    return out


def _commit(consumer, msg):
    try:
        consumer.commit()
        _jlog("kafka_commit", partition=getattr(msg, 'partition', None), offset=getattr(msg, 'offset', None))
    except Exception as e:  # noqa: BLE001
        _jlog("kafka_commit_fail", error=str(e))


def _build_model(MODEL_TYPE: str, config: Dict[str, Any], device, NUM_FEATURES: int, N_EXO_FEATURES: int, OUTPUT_SEQ_LEN: int):
    optimizer = "adam"
    schedule = False
    ss = False

    if MODEL_TYPE == "LSTM":
        HIDDEN_SIZE: int = int(env_var("HIDDEN_SIZE"))
        NUM_LAYERS: int = int(env_var("NUM_LAYERS"))
        config.update({"hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS})
        model = LSTM(input_size=NUM_FEATURES, n_exo_features=N_EXO_FEATURES, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SEQ_LEN, num_layers=NUM_LAYERS).to(device)
    elif MODEL_TYPE == "GRU":
        HIDDEN_SIZE = int(env_var("HIDDEN_SIZE"))
        NUM_LAYERS = int(env_var("NUM_LAYERS"))
        config.update({"hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS})
        model = GRU(input_size=NUM_FEATURES, n_exo_features=N_EXO_FEATURES, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SEQ_LEN, num_layers=NUM_LAYERS).to(device)
    elif MODEL_TYPE == "TETS":
        MODEL_DIM = int(env_var("MODEL_DIM"))
        NUM_HEADS = int(env_var("NUM_HEADS"))
        NUM_LAYERS = int(env_var("NUM_LAYERS"))
        FEEDFORWARD_DIM = int(env_var("FEEDFORWARD_DIM"))
        DROPOUT = float(env_var("DROPOUT"))
        optimizer = "adamw"
        schedule = True
        config.update({"model_dim": MODEL_DIM, "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS, "feedforward_dim": FEEDFORWARD_DIM, "dropout": DROPOUT})
        model = TETS(input_size=NUM_FEATURES, n_exo_features=N_EXO_FEATURES, output_size=OUTPUT_SEQ_LEN, model_dim=MODEL_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, feedforward_dim=FEEDFORWARD_DIM, dropout=DROPOUT).to(device)
    elif MODEL_TYPE == "TCN":
        LAYER_ARCHITECTURE: List[int] = [int(x) for x in env_var("LAYER_ARCHITECTURE").strip("[]").split(",")]
        KERNEL_SIZE = int(env_var("KERNEL_SIZE"))
        DROPOUT = float(env_var("DROPOUT"))
        config.update({"layer_architecture": LAYER_ARCHITECTURE, "kernel_size": KERNEL_SIZE, "dropout": DROPOUT})
        model = TCN(input_size=NUM_FEATURES, output_size=OUTPUT_SEQ_LEN, n_exo_features=N_EXO_FEATURES, layer_architecture=LAYER_ARCHITECTURE, kernel_size=KERNEL_SIZE, dropout=DROPOUT)
    elif MODEL_TYPE == "EncoderLSTM":
        HIDDEN_SIZE = int(env_var("HIDDEN_SIZE"))
        NUM_LAYERS = int(env_var("NUM_LAYERS"))
        ss = True
        config.update({"hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS})
        model = EncoderLSTM(input_size=NUM_FEATURES, n_exo_features=N_EXO_FEATURES, hidden_size=HIDDEN_SIZE, output_seq_len=OUTPUT_SEQ_LEN, num_layers=NUM_LAYERS).to(device)
    else:
        raise ValueError(f"{MODEL_TYPE} not supported")
    return model, optimizer, schedule, ss


def _train_parquet(df: pd.DataFrame, meta: Dict[str, Any]):
    MODEL_TYPE = env_var("MODEL_TYPE")
    experiment_name = env_var("EXPERIMENT_NAME", "Default")
    CONFIG_HASH = meta.get("config_hash")
    PREPROCESS_CONFIG = meta.get("preprocess_config")

    # Window / model parameters (defer feature counting until after ensuring target column)
    INPUT_SEQ_LEN = int(env_var("INPUT_SEQ_LEN"))
    OUTPUT_SEQ_LEN = int(env_var("OUTPUT_SEQ_LEN"))
    BATCH_SIZE = int(env_var("BATCH_SIZE", 32))
    EPOCHS = int(env_var("EPOCHS", 3))
    LEARNING_RATE = float(env_var("LEARNING_RATE", 1e-3))
    EARLY_STOPPING = env_var("EARLY_STOPPING", "true").lower() == "true"
    PATIENCE = int(env_var("PATIENCE", 10))

    # Ensure target column 'value' exists; create synthetic one if needed and HARD DROP original
    # to prevent feature duplication that leads to recurrent "Expected N got N+1" errors.
    original_target = None
    if 'value' not in df.columns:
        candidate_cols = [c for c in df.columns if c.lower() not in ("timestamp", "time")]  # quick filter
        numeric_candidates: List[str] = []
        for c in candidate_cols:
            try:
                if pd.api.types.is_numeric_dtype(df[c]):
                    numeric_candidates.append(c)
            except Exception:
                continue
        if not numeric_candidates:
            raise ValueError("Could not find numeric column to use as target when 'value' missing")
        original_target = numeric_candidates[0]
        df = df.copy()
        df['value'] = df[original_target]
        _jlog("target_fallback", chosen=original_target, note="Created synthetic 'value' column")
    # Attempt to drop original target if it still remains (defensive idempotent)
    if original_target and original_target != 'value' and original_target in df.columns:
        try:
            df.drop(columns=[original_target], inplace=True)
            _jlog("target_source_dropped", dropped=original_target)
        except Exception as drop_err:  # noqa: BLE001
            _jlog("target_source_drop_fail", dropped=original_target, error=str(drop_err))
    # Ensure only a single 'value' column present
    if list(df.columns).count('value') > 1:
        # Deduplicate by keeping the first occurrence
        dedup_cols = []
        seen_value = False
        for c in df.columns:
            if c == 'value':
                if seen_value:
                    continue
                seen_value = True
            dedup_cols.append(c)
        df = df[dedup_cols]
        _jlog("target_value_dedup", new_col_count=len(df.columns))

    # We will window the raw (post-target) DataFrame first, then infer NUM_FEATURES from X.shape to
    # eliminate any mismatch between model.input_size and actual tensor last dimension.
    pre_window_cols = list(df.columns)
    try:
        X, y, _scaler_obj = window_data(df, exo_features=None, input_len=INPUT_SEQ_LEN, output_len=OUTPUT_SEQ_LEN)
    except Exception as werr:  # noqa: BLE001
        _jlog("window_data_fail", error=str(werr), pre_window_columns=pre_window_cols)
        raise

    # Defensive: if model dimension mismatch has previously occurred (input.size(-1) != input_size)
    # it's almost always due to a stale duplicate feature (e.g., original target not dropped).
    # Verify the DataFrame column count equals X.shape[2].
    if X.shape[2] != len(df.columns):
        _jlog("feature_count_mismatch", df_cols=len(df.columns), X_last_dim=X.shape[2], columns=list(df.columns))
        # Attempt automatic repair: drop any duplicate 'value' or stray columns beyond the first len(df.columns)
        # (Simplistic repair: rebuild X/Y from a trimmed DataFrame.)
        df_repair = df.loc[:, ~df.columns.duplicated()].copy()
        if list(df_repair.columns).count('value') > 1:
            df_repair = df_repair.loc[:, [c for i,c in enumerate(df_repair.columns) if c != 'value' or i == list(df_repair.columns).index('value')]]
        try:
            X2, y2, _scaler_obj2 = window_data(df_repair, exo_features=None, input_len=INPUT_SEQ_LEN, output_len=OUTPUT_SEQ_LEN)
            if X2.shape[2] != X.shape[2]:
                X, y, _scaler_obj = X2, y2, _scaler_obj2
                _jlog("feature_mismatch_repaired", new_dim=X.shape[2], repaired_cols=list(df_repair.columns))
                df = df_repair
        except Exception as rep_err:  # noqa: BLE001
            _jlog("feature_mismatch_repair_fail", error=str(rep_err))

    # Derive feature counts directly from produced tensors
    NUM_FEATURES = int(X.shape[2])
    if 'value' not in df.columns:
        raise ValueError("'value' column missing after windowing phase")
    if NUM_FEATURES <= 1:
        raise ValueError(f"Not enough features to train model (NUM_FEATURES={NUM_FEATURES})")
    # Treat all but target as exogenous for now (simple heuristic)
    N_EXO_FEATURES = NUM_FEATURES - 1
    _jlog("feature_summary", num_features=NUM_FEATURES, n_exogenous=N_EXO_FEATURES, pre_window_columns=pre_window_cols, X_shape=list(X.shape), y_shape=list(y.shape))

    X_train, y_train = X, y  # (No validation split yet)
    X_test, y_test = X, y

    # Persist dummy scaler (placeholder for compatibility with downstream eval/inference)
    scaler_path = os.path.join(tempfile.gettempdir(), f"scaler_{int(time.time())}.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(_scaler_obj, f)

    train_loader, test_loader = prepare_data_loaders(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config: Dict[str, Any] = {
        "device": str(device),
        "input_size": NUM_FEATURES,
        "num_exogenous_features": N_EXO_FEATURES,
        "input_sequence_length": INPUT_SEQ_LEN,
        "output_sequence_length": OUTPUT_SEQ_LEN,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "early_stopping": EARLY_STOPPING,
        "model_type": MODEL_TYPE,
        "batch_size": BATCH_SIZE,
    }
    if EARLY_STOPPING:
        config['patience'] = PATIENCE

    model, optimizer, schedule, ss = _build_model(MODEL_TYPE, config, device, NUM_FEATURES, N_EXO_FEATURES, OUTPUT_SEQ_LEN)

    mlflow.set_experiment(experiment_name)
    mlflow.autolog()
    with mlflow.start_run(run_name=MODEL_TYPE, log_system_metrics=True) as run:
        run_id = run.info.run_id
        start_time = time.time()
        mlflow.log_params(config)
        if CONFIG_HASH:
            mlflow.log_param("config_hash", CONFIG_HASH)
        else:
            # Fallback default config hash so downstream eval/promotion has a grouping key
            CONFIG_HASH = os.environ.get("DEFAULT_CONFIG_HASH", "default_cfg")
            mlflow.log_param("config_hash", CONFIG_HASH)
            _jlog("config_hash_defaulted", run_id=run_id, model_type=MODEL_TYPE, config_hash=CONFIG_HASH)
        if PREPROCESS_CONFIG:
            mlflow.log_text(json.dumps(PREPROCESS_CONFIG, sort_keys=True), "preprocess/preprocess_config.json")
        identifier = os.environ.get("IDENTIFIER")
        _last_run_context.update({'run_id': run_id, 'model_type': MODEL_TYPE, 'config_hash': CONFIG_HASH, 'identifier': identifier, 'start_time': start_time})
        _jlog("train_start", run_id=run_id, model_type=MODEL_TYPE, config_hash=CONFIG_HASH, identifier=identifier, duration_ms=0)

        # Publish training start metadata
        try:
            producer = create_producer()
            topic = os.environ.get("PRODUCER_TOPIC") or "model-training"
            produce_message(producer, topic, {"operation": f"Training Started: {MODEL_TYPE}", "status": "RUNNING", "experiment": experiment_name, "run_name": MODEL_TYPE, "config_hash": CONFIG_HASH})
        except Exception as pe:  # noqa: BLE001
            publish_error(producer, f"DLQ-{os.environ.get('PRODUCER_TOPIC','model-training')}", "Publish training start", "Failure", str(pe), {"model_type": MODEL_TYPE, "config_hash": CONFIG_HASH})

        mlflow.log_artifact(scaler_path, artifact_path="scaler")

        model = train(model, train_loader, test_loader, epochs=EPOCHS, optimizer_type=optimizer, scheduled_learning=schedule, scheduled_sampling=ss, lr=LEARNING_RATE, criterion="mse", max_grad_norm=1.0, device=device, early_stopping=True, patience=PATIENCE)

    # Log model artifact
        try:
            # Fallback manual weights save (ensures directory appears even if mlflow hook has issues)
            try:
                import torch as _torch
                import tempfile as _tmp
                tmpd = _tmp.mkdtemp()
                weights_path = os.path.join(tmpd, "weights.pt")
                _torch.save(model.state_dict(), weights_path)
                mlflow.log_artifact(weights_path, artifact_path=MODEL_TYPE)
                _jlog("model_weights_logged", run_id=run_id, model_type=MODEL_TYPE, file="weights.pt")
            except Exception as fw:  # noqa: BLE001
                _jlog("model_weights_log_fail", run_id=run_id, model_type=MODEL_TYPE, error=str(fw))
            mlflow.pytorch.log_model(pytorch_model=model, artifact_path=MODEL_TYPE, input_example=X_train[:1], code_paths=["ml_models.py"])  # type: ignore
            _jlog("model_logged", run_id=run_id, model_type=MODEL_TYPE)
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                arts = client.list_artifacts(run_id, path="")
                _jlog("artifact_root_list", run_id=run_id, items=[a.path for a in arts])
                model_dir_list = client.list_artifacts(run_id, path=MODEL_TYPE)
                _jlog("artifact_model_list", run_id=run_id, model_type=MODEL_TYPE, items=[a.path for a in model_dir_list])
            except Exception as le:  # noqa: BLE001
                _jlog("artifact_list_fail", run_id=run_id, error=str(le))
        except Exception as e:  # noqa: BLE001
            _jlog("model_log_fail", error=str(e), run_id=run_id, model_type=MODEL_TYPE)
            raise

        # Publish training success message AFTER artifact logging so evaluators only see successes with artifacts
        try:
            success_payload = {
                "operation": f"Trained: {MODEL_TYPE}",
                "status": "SUCCESS",
                "experiment": experiment_name,
                "run_name": MODEL_TYPE,
                "config_hash": CONFIG_HASH,
                "run_id": run_id,
            }
            produce_message(producer, os.environ.get("PRODUCER_TOPIC") or "model-training", success_payload, key=f"trained-{MODEL_TYPE}")
            _jlog("train_success_publish", run_id=run_id, model_type=MODEL_TYPE, config_hash=CONFIG_HASH)
        except Exception as pe:  # noqa: BLE001
            publish_error(producer, f"DLQ-{os.environ.get('PRODUCER_TOPIC','model-training')}", "Publish training success", "Failure", str(pe), {"model_type": MODEL_TYPE, "config_hash": CONFIG_HASH, "run_id": run_id})
            _jlog("train_success_publish_fail", error=str(pe), run_id=run_id, model_type=MODEL_TYPE)

        # Loss curve
        try:
            import matplotlib.pyplot as plt
            client = mlflow.tracking.MlflowClient()
            train_hist = client.get_metric_history(run_id, "train_loss")
            test_hist = client.get_metric_history(run_id, "test_loss")
            if train_hist:
                plt.figure(figsize=(8, 4))
                plt.plot([m.step for m in train_hist], [m.value for m in train_hist], label="train_loss")
                if test_hist:
                    plt.plot([m.step for m in test_hist], [m.value for m in test_hist], label="test_loss")
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.title("Loss Curve")
                plt.legend()
                plt.tight_layout()
                mlflow.log_figure(plt.gcf(), "plots/loss_curve.png")
                plt.close()
                _jlog("loss_curve_logged")
        except Exception as e:  # noqa: BLE001
            _jlog("loss_curve_fail", error=str(e))

    end_time = time.time()
    duration_ms = int((end_time - _last_run_context.get('start_time', end_time)) * 1000)
    _jlog("train_complete", run_id=_last_run_context.get('run_id'), model_type=_last_run_context.get('model_type'), config_hash=_last_run_context.get('config_hash'), identifier=_last_run_context.get('identifier'), duration_ms=duration_ms)


def message_handler():
    _jlog("worker_start")
    GATEWAY_URL = env_var("GATEWAY_URL")
    global consumer  # commit access
    while True:
        msg = message_queue.get()
        try:
            claim = msg.value
            bucket = claim.get("bucket")
            object_key = claim.get("object") or claim.get("object_key")
            op = claim.get("operation")
            _jlog("claim_check", bucket=bucket, object_key=object_key, operation=op)
        except Exception as e:  # noqa: BLE001
            _jlog("claim_parse_error", error=str(e))
            message_queue.task_done()
            continue

        if op in ("post: train data", None) and bucket and object_key:
            try:
                _jlog("download_start", object_key=object_key)
                parquet_bytes = get_file(GATEWAY_URL, bucket, object_key)
                table = pq.read_table(source=parquet_bytes)
                df = table.to_pandas()
                schema = pq.read_schema(parquet_bytes)
                meta = _extract_meta(schema)
                _jlog("download_done", rows=len(df), cols=len(df.columns), config_hash=meta.get('config_hash'))
            except Exception as e:  # noqa: BLE001
                _jlog("download_fail", error=str(e), object_key=object_key)
                _commit(consumer, msg)
                message_queue.task_done()
                continue

            # Duplicate suppression: skip if this MODEL_TYPE+config_hash already processed
            model_type = os.environ.get("MODEL_TYPE", "UNKNOWN")
            cfg_hash = meta.get('config_hash') or os.environ.get("DEFAULT_CONFIG_HASH") or "default_cfg"
            signature = (model_type, cfg_hash)
            if SKIP_DUPLICATE_CONFIGS and signature in _processed_config_models:
                _jlog("train_skip_duplicate", model_type=model_type, config_hash=cfg_hash, object_key=object_key)
                _commit(consumer, msg)
                message_queue.task_done()
                continue

            failures_key = object_key
            if failures_key not in _failure_counts:
                _failure_counts[failures_key] = 0

            try:
                _train_parquet(df, meta)
                _jlog("train_complete", object_key=object_key, config_hash=meta.get('config_hash'))
                # Mark as processed (after success path)
                if SKIP_DUPLICATE_CONFIGS and signature not in _processed_config_models:
                    _processed_config_models.add(signature)
                    # Simple size cap eviction (FIFO order not strictly kept but adequate)
                    if len(_processed_config_models) > MAX_PROCESSED_CACHE:
                        # Pop an arbitrary element to keep memory bounded
                        _processed_config_models.pop()
                _commit(consumer, msg)
            except Exception as e:  # noqa: BLE001
                _failure_counts[failures_key] += 1
                fc = _failure_counts[failures_key]
                _jlog("train_error", error=str(e), object_key=object_key)
                if fc >= FAILURE_MAX_RETRIES:
                    try:
                        producer = create_producer()
                        publish_error(producer, f"DLQ-{os.environ.get('PRODUCER_TOPIC','model-training')}", "train", "Failure", str(e), {"object_key": object_key, "failures": fc})
                        _jlog("train_dlq", object_key=object_key, failures=fc)
                        _commit(consumer, msg)
                    except Exception as pe:  # noqa: BLE001
                        _jlog("dlq_publish_fail", error=str(pe))
                else:
                    _jlog("train_retry_defer", object_key=object_key, attempt=fc)
            finally:
                message_queue.task_done()
        else:
            _jlog("claim_unhandled", claim=claim)
            _commit(consumer, msg)
            message_queue.task_done()


TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]

trims = os.environ.get("TRIMS", "[]").strip("[]").split(",")
TRIMS = [item.strip() for item in trims if item.strip()]

message_queue = queue.Queue()
worker_thread = threading.Thread(target=message_handler, daemon=True)
worker_thread.start()

from random import randint  # debug unique group ids
# Honor explicit CONSUMER_GROUP_ID if provided; else randomize to avoid offset collisions during dev
_cg_override = os.environ.get("CONSUMER_GROUP_ID")
CONSUMER_GROUP_ID = _cg_override if _cg_override else f"CONSUMER_GROUP_ID{randint(0, 999)}"
consumer = create_consumer(os.environ.get("CONSUMER_TOPIC"), CONSUMER_GROUP_ID)
consume_messages(consumer, callback)
