import os
import queue
import threading
import tempfile
import pickle
import mlflow  # type: ignore
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pyarrow.parquet as pq
from typing import Any, List
import json
import time
from client_utils import get_file
from data_utils import check_uniform, subset_scaler
from kafka_utils import create_producer, create_consumer, produce_message, consume_messages, publish_error
from models import ProphetMultiFeatureModel, StatsForecastMultiFeatureModel

def env_var(var: str, default: Any=None) -> str:
    temp = os.environ.get(var, default)
    if temp is None:
        raise TypeError(f"Environment variable, {var}, not defined")
    else:
        return temp
    
def estimate_season_length(td: pd.Timedelta) -> int:
    """
    Infer smallest likely season length (in steps) greater than the data periodicity.
    
    Examples:
        1 min  -> 1440 (daily seasonality)
        5 min  -> 288  (daily seasonality)
        1 hour -> 24   (daily seasonality)
        1 day  -> 7    (weekly seasonality)
        1 week -> 52   (yearly seasonality)
    """
    seconds = td.total_seconds()

    # Define "likely" real-world seasonal cycles in seconds
    cycles = {
        "daily":   24 * 3600,
        "weekly":  7  * 24 * 3600,
        "yearly":  365 * 24 * 3600
    }

    # Find the smallest cycle > periodicity
    for name, cycle_seconds in cycles.items():
        if cycle_seconds > seconds:
            return int(round(cycle_seconds / seconds))

    # If the periodicity is larger than yearly, default to 1
    return 1

def _jlog(event: str, **extra):
    base = {"service": "nonml_train", "event": event}
    for k, v in extra.items():
        if v is not None:
            base[k] = v
    print(json.dumps(base))


def callback(message):
    _jlog("kafka_receive", key=message.key, partition=getattr(message, 'partition', None), offset=getattr(message, 'offset', None))
    message_queue.put(message)

FAILURE_MAX_RETRIES = int(os.environ.get("FAILURE_MAX_RETRIES", "3"))
_failure_counts = {}
# Duplicate suppression controls (avoid infinite loops on same config hash)
SKIP_DUPLICATE_CONFIGS = os.environ.get("SKIP_DUPLICATE_CONFIGS", "1").lower() in {"1","true","yes"}
from typing import Set, Tuple
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
                md = _extract_meta(schema)
                scaler = md.get('scaler_object')
                if TRIMS:
                    scaler = subset_scaler(scaler, df.columns.to_list(), TRIMS) if scaler else None
                    df.drop(columns=df.columns.difference(TRIMS + TIME_FEATURES), inplace=True)
                _jlog("download_done", rows=len(df), cols=len(df.columns), config_hash=md.get('config_hash'))
            except Exception as e:  # noqa: BLE001
                _jlog("download_fail", error=str(e), object_key=object_key)
                fc = _failure_counts.get(object_key, 0) + 1
                _failure_counts[object_key] = fc
                if fc >= FAILURE_MAX_RETRIES:
                    try:
                        producer = create_producer()
                        publish_error(producer, f"DLQ-{os.environ.get('PRODUCER_TOPIC','training-data')}", "nonml_download", "Failure", str(e), {"object_key": object_key})
                        _commit(consumer, msg)
                    except Exception:
                        pass
                message_queue.task_done()
                continue

            # Duplicate suppression: skip if this model+config already processed successfully
            model_type = os.environ.get("MODEL_TYPE", "UNKNOWN")
            cfg_hash = md.get('config_hash') or os.environ.get("DEFAULT_CONFIG_HASH") or "default_cfg"
            signature = (model_type, cfg_hash)
            if SKIP_DUPLICATE_CONFIGS and signature in _processed_config_models:
                _jlog("train_skip_duplicate", model_type=model_type, config_hash=cfg_hash, object_key=object_key)
                _commit(consumer, msg)
                message_queue.task_done()
                continue

            try:
                _jlog("train_start", object_key=object_key, config_hash=md.get('config_hash'))
                main(df, scaler, preprocess_meta=md)
                _jlog("train_complete", object_key=object_key, config_hash=md.get('config_hash'))
                _failure_counts.pop(object_key, None)
                if SKIP_DUPLICATE_CONFIGS and signature not in _processed_config_models:
                    _processed_config_models.add(signature)
                    if len(_processed_config_models) > MAX_PROCESSED_CACHE:
                        _processed_config_models.pop()
                _commit(consumer, msg)
            except Exception as e:  # noqa: BLE001
                _jlog("train_error", error=str(e), object_key=object_key)
                fc = _failure_counts.get(object_key, 0) + 1
                _failure_counts[object_key] = fc
                if fc >= FAILURE_MAX_RETRIES:
                    _jlog("train_dlq", object_key=object_key, failures=fc)
                    try:
                        producer = create_producer()
                        publish_error(producer, f"DLQ-{os.environ.get('PRODUCER_TOPIC','training-data')}", "nonml_train", "Failure", str(e), {"object_key": object_key, "config_hash": md.get('config_hash')})
                        _commit(consumer, msg)
                    except Exception:
                        pass
                else:
                    _jlog("train_retry_defer", object_key=object_key, attempt=fc)
            finally:
                message_queue.task_done()
        else:
            _jlog("claim_unhandled", claim=claim)
            message_queue.task_done()

def main(df: pd.DataFrame, scaler, experiment_name: str = "NonML", preprocess_meta=None):
    OUTPUT_SEQ_LEN: int = int(os.environ.get("OUTPUT_SEQ_LEN", "1"))
    MODEL_TYPE: str = env_var("MODEL_TYPE")
    SCALER_TYPE = scaler.__class__.__name__ # If negative values necessitate changes for certain models
    TRAIN_TEST_SPLIT = float(os.environ.get("TRAIN_TEST_SPLIT", "0.8"))
    if TRAIN_TEST_SPLIT <= 0 or TRAIN_TEST_SPLIT >= 1:
        TRAIN_TEST_SPLIT = 0.8  # sane fallback
    
    # === MLflow Logging ===
    mlflow.set_experiment(experiment_name)

    run_name = MODEL_TYPE

    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        CONFIG_HASH = None
        if preprocess_meta:
            CONFIG_HASH = preprocess_meta.get('config_hash')
            PREPROCESS_CONFIG = preprocess_meta.get('preprocess_config')
            if CONFIG_HASH:
                mlflow.log_param("config_hash", CONFIG_HASH)
            if PREPROCESS_CONFIG:
                mlflow.log_text(json.dumps(PREPROCESS_CONFIG, sort_keys=True), "preprocess/preprocess_config.json")
        # DEBUG/Redundant: ensure config_hash is always set as a param even if above block skipped
        if CONFIG_HASH:
            try:
                mlflow.log_param("config_hash", CONFIG_HASH)
            except Exception:
                pass
        # Diagnostic: verify config_hash param persisted (Prophet was missing from eval searches)
        if CONFIG_HASH:
            try:
                info_before = mlflow.get_run(run.info.run_id)
                value = info_before.data.params.get("config_hash") if info_before else None
                if not value:
                    _jlog("prophet_param_missing", run_id=run.info.run_id, expected=CONFIG_HASH)
                    try:
                        mlflow.log_param("config_hash", CONFIG_HASH)
                    except Exception as re_log_err:  # noqa: BLE001
                        _jlog("prophet_param_relog_fail", run_id=run.info.run_id, error=str(re_log_err))
                    try:
                        info_after = mlflow.get_run(run.info.run_id)
                        final_val = info_after.data.params.get("config_hash") if info_after else None
                        _jlog("prophet_param_verify", run_id=run.info.run_id, final=final_val)
                    except Exception as verify_err:  # noqa: BLE001
                        _jlog("prophet_param_verify_fail", run_id=run.info.run_id, error=str(verify_err))
                else:
                    _jlog("prophet_param_present", run_id=run.info.run_id, value=value)
            except Exception as check_err:  # noqa: BLE001
                _jlog("prophet_param_check_fail", run_id=run.info.run_id, error=str(check_err))
        # Standard params for downstream eval filtering
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("train_test_split", TRAIN_TEST_SPLIT)
        identifier = os.environ.get("IDENTIFIER")
        start_time = time.time()
        _jlog(
            "train_start",
            run_id=run.info.run_id,
            model_type=MODEL_TYPE,
            config_hash=CONFIG_HASH,
            identifier=identifier,
            duration_ms=0,
        )
        # mlflow.autolog() will not work with current dependencies

        # Get data periodicity
        timedelta = check_uniform(df)
        offset = to_offset(timedelta).freqstr # type: ignore

        # Change datetime index to "ds" column for StatsForecast and Prophet
        df.index.rename("ds", inplace=True)
        df = df.reset_index()

        # Get feature columns (excluding ds and unique_id)
        feature_columns = [col for col in df.columns if col not in (["ds", "unique_id"]+TIME_FEATURES)]
        
        print(f"Feature columns: {feature_columns}")
        print(f"Non-Feature columns: {df.columns.difference(feature_columns, sort=False).to_list()}")

        # Save scaler to a temporary file because MLflow can only save artifacts from files
        scaler_path = os.path.join(tempfile.gettempdir(), "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        # Log scaler as an artifact
        mlflow.log_artifact(scaler_path, artifact_path="scaler")

        if MODEL_TYPE == "PROPHET":
            # One could add a hyperparameter grid search here but it would be very slow
            # and since Data Scientist input is needed to choose sensible ranges anyway,
            # might as well just set them directly.

            prophet_params = {
                "growth": os.environ.get("GROWTH", "linear"),
                "n_changepoints": int(os.environ.get("N_CHANGEPOINTS", "25")),
                "changepoint_range": float(os.environ.get("CHANGEPOINT_RANGE", "0.8")),
                "yearly_seasonality": os.environ.get("YEARLY_SEASONALITY", "auto"),
                "weekly_seasonality": os.environ.get("WEEKLY_SEASONALITY", "auto"),
                "daily_seasonality": os.environ.get("DAILY_SEASONALITY", "auto"),
                "seasonality_mode": os.environ.get("SEASONALITY_MODE", "additive"),
                "seasonality_prior_scale": float(os.environ.get("SEASONALITY_PRIOR_SCALE", "10")),
                "holidays_prior_scale": float(os.environ.get("HOLIDAYS_PRIOR_SCALE", "10")),
                "changepoint_prior_scale": float(os.environ.get("CHANGEPOINT_PRIOR_SCALE", "0.05")),
                "country": os.environ.get("COUNTRY", "US") # for built-in holiday effects
            }
            
            # Log all Prophet parameters
            mlflow.log_params({f"{k}": v for k, v in prophet_params.items()})
            
            # Create and fit multi-feature Prophet model
            multi_prophet = ProphetMultiFeatureModel()

            # Temporal split
            split_idx = int(len(df) * TRAIN_TEST_SPLIT)
            if split_idx <= 1 or split_idx >= len(df):
                split_idx = max(2, int(len(df) * 0.8))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            # Fit ONLY on train portion for honest test metrics
            multi_prophet.fit(train_df, feature_columns, prophet_params)

            # Predictions for train & test horizons
            import math as _math
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            # Helper to extract only feature columns (exclude *_yhat_lower/upper)
            def _extract_pred(pred_df):
                keep = [c for c in pred_df.columns if c in feature_columns]
                return pred_df[keep]

            metrics_ok = False
            try:
                _jlog("metrics_compute_start", run_id=run.info.run_id, model_type=MODEL_TYPE, config_hash=CONFIG_HASH)
                # Get predictions (index becomes datetime 'ds')
                train_pred_raw = multi_prophet.predict(None, train_df[["ds"]])
                test_pred_raw = multi_prophet.predict(None, test_df[["ds"]]) if len(test_df) > 0 else None

                def _align_and_metrics(true_df, pred_raw):
                    # true_df has columns feature_columns plus ds
                    actual = true_df[['ds'] + feature_columns].set_index('ds')
                    pred = _extract_pred(pred_raw)
                    # Ensure prediction index is datetime; if not, set index to ds column
                    if 'ds' in pred.columns and not isinstance(pred.index, pd.DatetimeIndex):
                        # rare path; normally index already set to ds in predict
                        pred = pred.set_index('ds')
                    # Intersect indexes
                    common_idx = actual.index.intersection(pred.index)
                    if len(common_idx) == 0:
                        raise ValueError("No overlapping timestamps between actual and prediction for metrics computation")
                    a_np = actual.loc[common_idx].to_numpy()
                    p_np = pred.loc[common_idx][feature_columns].to_numpy()
                    # Compute metrics across all features (multi-output regression)
                    mse_v = mean_squared_error(a_np, p_np)
                    mae_v = mean_absolute_error(a_np, p_np)
                    rmse_v = _math.sqrt(mse_v)
                    return mse_v, rmse_v, mae_v

                train_mse, train_rmse, train_mae = _align_and_metrics(train_df[['ds'] + feature_columns], train_pred_raw)
                mlflow.log_metrics({"train_mse": train_mse, "train_rmse": train_rmse, "train_mae": train_mae})

                if test_pred_raw is not None and len(test_df) > 0:
                    test_mse, test_rmse, test_mae = _align_and_metrics(test_df[['ds'] + feature_columns], test_pred_raw)
                    mlflow.log_metrics({"test_mse": test_mse, "test_rmse": test_rmse, "test_mae": test_mae})
                    _jlog("metrics_logged", run_id=run.info.run_id, model_type=MODEL_TYPE, config_hash=CONFIG_HASH, test_rmse=test_rmse, test_mae=test_mae, test_mse=test_mse)
                else:
                    _jlog("metrics_skipped", reason="no_test_rows", run_id=run.info.run_id, model_type=MODEL_TYPE, config_hash=CONFIG_HASH)
                metrics_ok = True
            except Exception as me:  # noqa: BLE001
                _jlog("metrics_fail", error=str(me), run_id=run.info.run_id, model_type=MODEL_TYPE, config_hash=CONFIG_HASH)
                # Do NOT raise; allow model logging & success publish so evaluation can still consider baseline
                metrics_ok = False

            # Log the bundled model AFTER metrics so success implies metrics presence
            try:
                # IMPORTANT: use artifact_path (not name) so artifacts are stored under a folder
                # matching the model_type (e.g. PROPHET) – evaluation filters runs by checking
                # client.list_artifacts(run_id, path=model_type). Previously we used the wrong
                # keyword (name=) so no PROPHET folder was created and eval skipped this run.
                mlflow.pyfunc.log_model(
                    artifact_path=MODEL_TYPE,
                    python_model=multi_prophet,
                    code_paths=["models.py"]
                )
                _jlog("model_logged", run_id=run.info.run_id, model_type=MODEL_TYPE, config_hash=CONFIG_HASH, metrics_ok=metrics_ok)
            except Exception as le:  # noqa: BLE001
                _jlog("model_log_fail", error=str(le), run_id=run.info.run_id, model_type=MODEL_TYPE, config_hash=CONFIG_HASH, metrics_ok=metrics_ok)
                # Still proceed; artifacts missing will cause eval to skip if unusable

        elif MODEL_TYPE in ["AUTOARIMA", "AUTOETS", "AUTOTHETA", "AUTOMFLES", "AUTOTBATS"]:

            DOWNSAMPLING = os.environ.get("DOWNSAMPLING", "0")
            
            if DOWNSAMPLING != "0":
                try:
                    downsampling = pd.Timedelta(DOWNSAMPLING)
                    print(f"Before downsampling: {df.shape[0]} rows \n{df['ds'].head(3)}")
                    df.set_index(["ds"], inplace=True)
                    df = df.resample(downsampling).mean() # Could add other aggregation methods (e.g., sum, max)
                    df.reset_index(inplace=True)
                    print(f"After downsampling: {df.shape[0]} rows \n{df['ds'].head(3)}")
                except ValueError:
                    raise ValueError(f"Invalid DOWNSAMPLING value: {DOWNSAMPLING}. Must be a valid pandas Timedelta string.")

            # Add dummy unique_id column because StatsForecast requires it
            df["unique_id"] = "1"

            sl_env = os.environ.get("SEASON_LENGTH", "0")
            sl: List[int] = [int(x) for x in sl_env.strip("[]").split(",")]

            if sl[0] == 0:
                SEASON_LENGTH: List[int] = [estimate_season_length(timedelta)]
            else:
                SEASON_LENGTH = sl

            # Collect StatsForecast parameters
            statsforecast_params = {
                "model_type": MODEL_TYPE,
                "output_sequence_length": OUTPUT_SEQ_LEN,
                "season_length": SEASON_LENGTH,
                "downsampling": DOWNSAMPLING,
                "frequency": offset,
            }
            
            # Log all StatsForecast parameters
            mlflow.log_params({f"{k}": v for k, v in statsforecast_params.items()})

            tf = df[df.columns.difference(["ds", "unique_id"] + TIME_FEATURES, sort=False)]
            ds = df["ds"]
            # Create and fit multi-feature StatsForecast model
            print(f"\ndf shape: {df.shape}, df shape: {df.shape}")
            print(f"Time range: {ds.iloc[0]} to {ds.iloc[-1]}")
            print(f"Number of values less than or equal to zero in each column:\n{(tf <= 0).sum()}")
            print(f"Maximum value:\n{tf.max()}\nMinimum value:\n{tf.min()}\n")
            multi_statsforecast = StatsForecastMultiFeatureModel()
            multi_statsforecast.fit(df, feature_columns, statsforecast_params, TIME_FEATURES)
            
            # Log the bundled model
            # Same artifact_path reasoning as in PROPHET branch – ensure eval finds artifacts.
            mlflow.pyfunc.log_model(
                artifact_path=MODEL_TYPE,
                python_model=multi_statsforecast,
                # signature=multi_statsforecast.get_signature(), # Providing a signature or input example is bait, it's not suited for custom model wrappers
                code_paths=["models.py"]
            )
            
        else:
            raise ValueError(f"{MODEL_TYPE} not supported")

        # Publish success (inside run so run_id always present)
        try:
            producer = create_producer()
            topic = os.environ.get("PRODUCER_TOPIC") or "model-training"
            success_payload = {
                "operation": f"Trained: {MODEL_TYPE}",
                "status": "SUCCESS",
                "experiment": experiment_name,
                "run_name": run_name,
                "config_hash": CONFIG_HASH,
                "run_id": run.info.run_id,
            }
            produce_message(producer, topic, success_payload, key=f"trained-{MODEL_TYPE}")
            _jlog("train_success_publish", run_id=run.info.run_id, model_type=MODEL_TYPE, config_hash=CONFIG_HASH)
        except Exception as pe:  # noqa: BLE001
            try:
                publish_error(create_producer(), f"DLQ-{os.environ.get('PRODUCER_TOPIC','model-training')}", "Publish training success", "Failure", str(pe), {"model_type": MODEL_TYPE, "config_hash": CONFIG_HASH})
            except Exception:
                pass
            _jlog("train_success_publish_fail", error=str(pe), model_type=MODEL_TYPE, config_hash=CONFIG_HASH)
        run_id_var = run.info.run_id

    end_time = time.time()
    duration_ms = int((end_time - start_time) * 1000)
    _jlog(
        "train_complete",
        run_id=locals().get('run_id_var'),
        model_type=MODEL_TYPE,
        config_hash=CONFIG_HASH,
        identifier=identifier,
        duration_ms=duration_ms,
    )
    print("✅ Model logged to MLflow")


TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]

trims = os.environ.get("TRIMS", "[]").strip("[]").split(",")
TRIMS: List = [item.strip().strip('"') for item in trims if item.strip().strip('"')]

message_queue = queue.Queue()
worker_thread = threading.Thread(target=message_handler, daemon=True)
worker_thread.start()

from random import randint
# Honor explicit CONSUMER_GROUP_ID if provided, else randomize to avoid accidental offset reuse
CG_OVERRIDE = os.environ.get("CONSUMER_GROUP_ID")
CONSUMER_GROUP_ID = CG_OVERRIDE if CG_OVERRIDE else f"CONSUMER_GROUP_ID{randint(0, 999)}"
consumer = create_consumer(env_var("CONSUMER_TOPIC"), CONSUMER_GROUP_ID)
consume_messages(consumer, callback)

