import os, json, time, traceback, threading, tempfile, datetime
from typing import Dict, Any, Optional, List, Set
import mlflow
import pandas as pd
from kafka_utils import create_consumer, create_producer, produce_message, publish_error
from dateutil import tz
import boto3

# Version marker for deployment verification
EVAL_VERSION = "eval_v20251002_01"

# Structured logging helper

def jlog(event: str, **extra):
    base = {"service": "eval", "event": event, "ts": datetime.datetime.utcnow().isoformat() + "Z", "version": EVAL_VERSION}
    base.update({k: v for k, v in extra.items() if v is not None})
    print(json.dumps(base), flush=True)

# Environment
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
MODEL_TRAINING_TOPIC = os.environ.get("MODEL_TRAINING_TOPIC", "model-training")
MODEL_SELECTED_TOPIC = os.environ.get("MODEL_SELECTED_TOPIC", "model-selected")
GROUP_ID = os.environ.get("EVAL_GROUP_ID", "eval-promoter")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MINIO_PROMOTION_BUCKET = os.environ.get("PROMOTION_BUCKET", "model-promotion")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://fastapi-app:8000")
IDENTIFIER_FALLBACK = os.environ.get("IDENTIFIER", "")
DLQ_TOPIC = os.environ.get("DLQ_MODEL_SELECTED", "DLQ-model-selected")
SCORE_WEIGHTS = {"rmse": 0.5, "mae": 0.3, "mse": 0.2}
LOOKBACK_RUNS = int(os.environ.get("LOOKBACK_RUNS", "50"))

# Expected model types for a pipeline config (comma separated env var). Default to GRU,LSTM,PROPHET.
EXPECTED_MODEL_TYPES: Set[str] = set([m.strip().upper() for m in os.environ.get("EXPECTED_MODEL_TYPES", "GRU,LSTM,PROPHET").split(",") if m.strip()])

# Retry controls to mitigate race where freshly finished runs (e.g., PROPHET) are not yet returned by MLflow search
PROMOTION_SEARCH_RETRIES = int(os.environ.get("PROMOTION_SEARCH_RETRIES", "3"))
PROMOTION_SEARCH_DELAY_SEC = float(os.environ.get("PROMOTION_SEARCH_DELAY_SEC", "2"))

# Track completion: config_hash -> set(model_types completed)
_completion_tracker: Dict[str, Set[str]] = {}

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
def _ensure_buckets():
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        required = [
            os.environ.get("MLFLOW_ARTIFACT_BUCKET", "mlflow"),
            MINIO_PROMOTION_BUCKET,
        ]
        existing = {b.get('Name') for b in s3.list_buckets().get('Buckets', [])}
        for b in required:
            if b not in existing:
                try:
                    s3.create_bucket(Bucket=b)
                    jlog("bucket_created", bucket=b)
                except Exception as ce:  # noqa: BLE001
                    jlog("bucket_create_fail", bucket=b, error=str(ce))
            else:
                jlog("bucket_exists", bucket=b)
    except Exception as e:  # noqa: BLE001
        jlog("bucket_ensure_error", error=str(e))

_ensure_buckets()
producer = create_producer()
consumer = create_consumer(MODEL_TRAINING_TOPIC, GROUP_ID)

# Simple MinIO gateway post helper (reusing FASTAPI gateway endpoint)
import requests

def upload_json(bucket: str, object_name: str, payload: Dict[str, Any]):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        tmp.write(json.dumps(payload, separators=(",", ":")).encode())
        tmp.flush()
        tmp.close()
        with open(tmp.name, "rb") as fh:
            files = {"file": (object_name, fh, "application/json")}
            r = requests.post(f"{GATEWAY_URL}/upload/{bucket}/{object_name}", files=files, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"Upload failed {r.status_code}: {r.text}")
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass


def compute_score(row: pd.Series) -> float:
    return (
        SCORE_WEIGHTS["rmse"] * float(row.get("metrics.test_rmse", float("inf"))) +
        SCORE_WEIGHTS["mae"] * float(row.get("metrics.test_mae", float("inf"))) +
        SCORE_WEIGHTS["mse"] * float(row.get("metrics.test_mse", float("inf")))
    )


def select_best(runs_df: pd.DataFrame) -> Optional[pd.Series]:
    if runs_df.empty:
        return None
    runs_df = runs_df.copy()
    runs_df["promotion_score"] = runs_df.apply(compute_score, axis=1)
    # lowest score wins; tie -> most recent start_time (mlflow stores unix ms)
    runs_df.sort_values(["promotion_score", "start_time"], ascending=[True, False], inplace=True)
    # Emit scoreboard log (trim huge DataFrames) before selection
    try:
        max_rows = int(os.environ.get("PROMOTION_SCOREBOARD_LIMIT", "50"))
        view = runs_df.head(max_rows)
        scoreboard = []
        for _, r in view.iterrows():
            st = r.get("start_time")
            # Convert pandas/np timestamp-like objects to ISO string for JSON
            try:
                if hasattr(st, 'to_pydatetime'):
                    st = st.to_pydatetime()
                if hasattr(st, 'isoformat'):
                    st = st.isoformat()
            except Exception:
                st = str(st)
            scoreboard.append({
                "run_id": r.get("run_id"),
                "model_type": r.get("params.model_type"),
                "test_rmse": r.get("metrics.test_rmse"),
                "test_mae": r.get("metrics.test_mae"),
                "test_mse": r.get("metrics.test_mse"),
                "score": r.get("promotion_score"),
                "start_time": st,
            })
        jlog("promotion_scoreboard", rows=len(view), scoreboard=scoreboard)
    except Exception as sb_err:  # noqa: BLE001
        jlog("promotion_scoreboard_fail", error=str(sb_err))
    return runs_df.iloc[0]


def promotion_payload(row: pd.Series, identifier: str, config_hash: str) -> Dict[str, Any]:
    run_id = row.get("run_id")
    model_type = row.get("params.model_type")
    experiment_id = row.get("experiment_id")
    experiment = mlflow.get_experiment(experiment_id).name if experiment_id else ""
    model_uri = f"runs:/{run_id}/{model_type}" if model_type else f"runs:/{run_id}"
    return {
        "identifier": identifier,
        "config_hash": config_hash,
        "run_id": run_id,
        "model_type": model_type,
        "experiment": experiment,
        "model_uri": model_uri,
        "rmse": row.get("metrics.test_rmse"),
        "mae": row.get("metrics.test_mae"),
        "mse": row.get("metrics.test_mse"),
        "score": row.get("promotion_score"),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "weights": SCORE_WEIGHTS,
    }


def process_training_message(msg_value: Dict[str, Any]):
    try:
        identifier = msg_value.get("identifier") or IDENTIFIER_FALLBACK
        config_hash = msg_value.get("config_hash")
        operation = msg_value.get("operation", "") or ""
        status = msg_value.get("status")
        # Only consider final success training messages
        if status != "SUCCESS" or not operation.startswith("Trained: "):
            jlog("promotion_ignore", reason="non_success_or_not_trained", operation=operation, status=status)
            return
        model_type = operation.replace("Trained: ", "").strip()
        if not config_hash:
            jlog("promotion_skip", reason="missing_config_hash", message=msg_value)
            return
        jlog("promotion_start", identifier=identifier, config_hash=config_hash, model_type=model_type)
        if config_hash not in _completion_tracker:
            _completion_tracker[config_hash] = set()
        if model_type:
            _completion_tracker[config_hash].add(model_type.upper())
        missing = sorted(list(EXPECTED_MODEL_TYPES - _completion_tracker[config_hash]))
        if missing:
            jlog("promotion_waiting_for_models", config_hash=config_hash, have=sorted(list(_completion_tracker[config_hash])), missing=missing, expected=sorted(list(EXPECTED_MODEL_TYPES)))
            return
        jlog("promotion_all_models_present", config_hash=config_hash, models=sorted(list(_completion_tracker[config_hash])))
        filter_string = f"params.config_hash = '{config_hash}'"
        # Expanded search: include ALL experiments so that heterogeneous model families (e.g. PROPHET in 'NonML' experiment
        # and GRU/LSTM in another) are all considered. Previously omission of experiment_ids could implicitly scope search
        # to an active/default experiment, excluding Prophet runs.
        from mlflow.tracking import MlflowClient  # local import to avoid module load at startup if mlflow unreachable
        _exp_client = MlflowClient()
        exp_ids = []
        experiments_meta = []
        # Compatibility: prefer search_experiments (newer MLflow), fallback to list_experiments if available
        try:
            if hasattr(_exp_client, "search_experiments"):
                _experiments = _exp_client.search_experiments()
            elif hasattr(_exp_client, "list_experiments"):
                _experiments = _exp_client.list_experiments()
            else:
                _experiments = []
            for e in _experiments:
                exp_ids.append(e.experiment_id)
                experiments_meta.append({"id": e.experiment_id, "name": getattr(e, "name", "")})
        except Exception as ee:  # noqa: BLE001
            jlog("promotion_experiment_enum_fail", error=str(ee))
        # Fallback: if enumeration failed, try known experiment names directly
        if not exp_ids:
            try:
                # 'Default' is almost always experiment_id '0'; 'NonML' may exist for Prophet
                named = [mlflow.get_experiment_by_name(n) for n in ["Default", "NonML"]]
                for ex in named:
                    if ex:
                        exp_ids.append(ex.experiment_id)
                        experiments_meta.append({"id": ex.experiment_id, "name": ex.name})
            except Exception as en_err:  # noqa: BLE001
                jlog("promotion_experiment_named_lookup_fail", error=str(en_err))
        if not exp_ids:
            jlog("promotion_no_experiments_found", config_hash=config_hash)
            return
        try:
            jlog("promotion_search_experiments", config_hash=config_hash, experiments=experiments_meta)
        except Exception:
            pass
        attempt = 0
        missing_model_types: Set[str] = set()
        while True:
            runs_df = mlflow.search_runs(experiment_ids=exp_ids, filter_string=filter_string, max_results=LOOKBACK_RUNS, output_format="pandas")  # type: ignore
            present = set()
            if not runs_df.empty:
                mt_col = [c for c in runs_df.columns if c == "params.model_type"]
                if mt_col:
                    present = set(runs_df[mt_col[0]].dropna().str.upper().tolist())
            missing_model_types = EXPECTED_MODEL_TYPES - present
            if runs_df.empty or missing_model_types:
                if attempt < PROMOTION_SEARCH_RETRIES - 1:
                    jlog("promotion_search_retry_wait", config_hash=config_hash, attempt=attempt+1, missing=list(missing_model_types), delay=PROMOTION_SEARCH_DELAY_SEC)
                    time.sleep(PROMOTION_SEARCH_DELAY_SEC)
                    attempt += 1
                    continue
                if runs_df.empty:
                    jlog("promotion_no_runs", config_hash=config_hash, attempts=attempt+1)
                    return
                else:
                    jlog("promotion_partial_runs", config_hash=config_hash, present=list(present), still_missing=list(missing_model_types), attempts=attempt+1)
                    break
            break
        # DEBUG: emit all runs found prior to artifact filtering to diagnose missing model types (e.g. PROPHET)
        try:
            debug_runs = []
            for _, r in runs_df.iterrows():
                # Collect all params columns (they are prefixed with 'params.')
                param_cols = {c.replace('params.', ''): r.get(c) for c in runs_df.columns if c.startswith('params.')}
                debug_runs.append({
                    "run_id": r.get("run_id"),
                    "model_type": r.get("params.model_type"),
                    "config_hash": r.get("params.config_hash"),
                    "all_params": param_cols
                })
            jlog("promotion_runs_search", config_hash=config_hash, count=len(debug_runs), runs=debug_runs)
        except Exception as dbg_err:  # noqa: BLE001
            jlog("promotion_runs_search_fail", config_hash=config_hash, error=str(dbg_err))
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            valid_rows: List[int] = []
            for idx, row in runs_df.iterrows():
                r_id = row.get("run_id")
                mtype = row.get("params.model_type")
                if not r_id or not mtype:
                    continue
                try:
                    # Primary: look for artifacts under folder named after model_type
                    arts = client.list_artifacts(r_id, path=mtype)
                    has_named_folder = bool(arts)
                    has_any_artifact = has_named_folder
                    artifact_names = [a.path for a in arts]
                    # Fallback: if none under model_type, list root artifacts and accept if any exist
                    if not has_named_folder:
                        root_arts = client.list_artifacts(r_id)
                        if root_arts:
                            has_any_artifact = True
                            artifact_names.extend([a.path for a in root_arts if a.path not in artifact_names])
                    # Additional Prophet-specific heuristic: accept if any artifact path contains 'scaler' or 'preprocess'
                    if not has_named_folder and mtype.upper() == "PROPHET":
                        # Heuristic 1: any artifacts at all (already covered) OR metrics present in runs_df
                        if artifact_names:
                            has_any_artifact = True
                        else:
                            # Metrics-based fallback: if the run has test metrics columns populated, treat as valid
                            metric_cols = [c for c in runs_df.columns if c.startswith("metrics.test_")]
                            metrics_present = False
                            if metric_cols:
                                for mc in metric_cols:
                                    try:
                                        val = row.get(mc)
                                        if val is not None and val == val:  # not NaN
                                            metrics_present = True
                                            break
                                    except Exception:
                                        continue
                            if metrics_present:
                                has_any_artifact = True
                                artifact_names.append("__metrics_only__")
                    if has_any_artifact:
                        valid_rows.append(idx)
                        jlog("promotion_artifacts_ok", run_id=r_id, model_type=mtype, named_folder=has_named_folder, artifacts=artifact_names[:20])
                    else:
                        jlog("promotion_skip_run_no_artifacts", run_id=r_id, model_type=mtype)
                except Exception as le:  # noqa: BLE001
                    jlog("promotion_artifact_list_fail", run_id=r_id, model_type=mtype, error=str(le))
            if not valid_rows:
                jlog("promotion_no_valid_runs", config_hash=config_hash)
                return
            runs_df = runs_df.loc[valid_rows]
        except Exception as e:  # noqa: BLE001
            jlog("promotion_artifact_filter_error", error=str(e))
        best = select_best(runs_df)
        if best is None:
            jlog("promotion_no_selection", config_hash=config_hash)
            return
        payload = promotion_payload(best, identifier, config_hash)
        jlog("promotion_decision", **payload)
        ts = payload["timestamp"].replace(":", "-")
        history_obj = f"promotion-{ts}.json"
        base_path = f"{identifier or 'global'}/{config_hash}"
        upload_json(MINIO_PROMOTION_BUCKET, f"{base_path}/{history_obj}", payload)
        # Identifier / legacy scoped pointer (global/<config_hash>/..., global/current.json)
        upload_json(MINIO_PROMOTION_BUCKET, f"{identifier or 'global'}/current.json", payload)
        # Root-level canonical pointer for simplified autoload (new)
        try:
            upload_json(MINIO_PROMOTION_BUCKET, "current.json", payload)
            jlog("promotion_root_pointer_write", run_id=payload["run_id"], model_type=payload.get("model_type"), config_hash=config_hash)
        except Exception as root_ptr_err:  # noqa: BLE001
            jlog("promotion_root_pointer_fail", error=str(root_ptr_err))
        produce_message(producer, MODEL_SELECTED_TOPIC, payload, key="promotion")
        jlog("promotion_publish", run_id=payload["run_id"], config_hash=config_hash)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        jlog("promotion_error", error=str(e))
        publish_error(producer, DLQ_TOPIC, "promotion", "Failure", str(e), msg_value)


def main_loop():
    jlog("service_start", topic=MODEL_TRAINING_TOPIC)
    for msg in consumer:
        try:
            process_training_message(msg.value)
        except Exception:
            traceback.print_exc()
            jlog("message_error")

#############################
# FastAPI health endpoints  #
#############################
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import threading as _threading

app = FastAPI()
_ready = {"kafka": False, "mlflow": False}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    ready = all(_ready.values())
    code = 200 if ready else 503
    return JSONResponse(status_code=code, content={"status": "ready" if ready else "not_ready", "components": _ready})

def _init_readiness_checks():
    # Kafka readiness: if consumer assigned at least one partition eventually
    try:
        _ready["kafka"] = True  # simplified; consumer already created
    except Exception:
        _ready["kafka"] = False
    # MLflow readiness
    try:
        mlflow.search_runs(max_results=1)
        _ready["mlflow"] = True
    except Exception:
        _ready["mlflow"] = False

def _run_service():
    _init_readiness_checks()
    main_loop()

def start_background_loop():
    t = _threading.Thread(target=_run_service, daemon=True)
    t.start()

start_background_loop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
