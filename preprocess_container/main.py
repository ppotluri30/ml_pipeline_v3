"""Modular, idempotent preprocessing service with recipe hashing.

Features:
 - Controlled by environment variable toggles (backward-compatible defaults).
 - Deterministic canonical config JSON -> SHA256 config_hash.
 - Idempotent: if outputs with same config_hash already exist (sidecar meta.json), skip work and re-emit claim checks.
 - Embeds metadata (preprocess_config, config_hash) in Parquet file metadata.
 - Writes meta sidecar JSON per output.
 - Structured JSON logging.
 - DLQ on failures.
 - /healthz (always 200) & /readyz (validates MinIO reachability + basic bucket/object preconditions) via lightweight FastAPI app.
"""
from __future__ import annotations

import io
import json
import os
import time
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type  # type: ignore

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from client_utils import get_file, post_file
from kafka_utils import create_producer, produce_message, publish_error
from data_utils import (
    read_data,
    handle_nans,
    clip_outliers,
    scale_data,
    generate_lags,
    time_to_feature,
)

DEFAULTS = {
    "HANDLE_NANS": True,
    "NANS_THRESHOLD": 0.33,
    "NANS_KNN": 2,
    "NANS_DROP_ROWS": False,
    "CLIP_ENABLE": False,
    "CLIP_METHOD": "iqr",  # or percentile
    "CLIP_FACTOR": 1.5,
    "TIME_FEATURES_ENABLE": True,
    "LAGS_ENABLE": False,
    "LAGS_N": 0,
    "LAGS_STEP": 1,
    "SCALER": "MinMaxScaler",  # empty string for none
    "ADD_VAL": None,
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def build_active_config() -> Dict[str, Any]:
    cfg = {
        "handle_nans": {
            "enabled": _env_bool("HANDLE_NANS", DEFAULTS["HANDLE_NANS"]),
            "threshold": _env_float("NANS_THRESHOLD", DEFAULTS["NANS_THRESHOLD"]),
            "knn_neighbors": _env_int("NANS_KNN", DEFAULTS["NANS_KNN"]),
            "drop_rows": _env_bool("NANS_DROP_ROWS", DEFAULTS["NANS_DROP_ROWS"]),
        },
        "outliers": {
            "enabled": _env_bool("CLIP_ENABLE", DEFAULTS["CLIP_ENABLE"]),
            "method": os.environ.get("CLIP_METHOD", DEFAULTS["CLIP_METHOD"]).lower(),
            "factor": _env_float("CLIP_FACTOR", DEFAULTS["CLIP_FACTOR"]),
        },
        "time_features": {"enabled": _env_bool("TIME_FEATURES_ENABLE", DEFAULTS["TIME_FEATURES_ENABLE"])},
        "lags": {
            "enabled": _env_bool("LAGS_ENABLE", DEFAULTS["LAGS_ENABLE"]),
            "n_lags": _env_int("LAGS_N", DEFAULTS["LAGS_N"]),
            "step": _env_int("LAGS_STEP", DEFAULTS["LAGS_STEP"]),
        },
        "scaling": {
            "method": os.environ.get("SCALER", DEFAULTS["SCALER"]).strip(),
            "add_constant": os.environ.get("ADD_VAL", DEFAULTS["ADD_VAL"]),
        },
        # Salt allows forcing a new hash without altering actual transformations; tracked for lineage.
        "extra": {
            "hash_salt": os.environ.get("EXTRA_HASH_SALT", "")
        }
    }
    return cfg


def canonical_config_blob(active_cfg: Dict[str, Any]) -> Tuple[str, str]:
    # Deterministic JSON (sorted keys, no spaces)
    canonical = json.dumps(active_cfg, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(canonical.encode()).hexdigest()
    return canonical, h


def apply_pipeline(train: pd.DataFrame, test: pd.DataFrame, p: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # 1. Missing values
    if p["handle_nans"]["enabled"]:
        train = handle_nans(
            train,
            threshold=p["handle_nans"]["threshold"],
            window=p["handle_nans"]["knn_neighbors"],
            no_drop=not p["handle_nans"]["drop_rows"],
        )
        test = handle_nans(
            test,
            threshold=p["handle_nans"]["threshold"],
            window=p["handle_nans"]["knn_neighbors"],
            no_drop=not p["handle_nans"]["drop_rows"],
        )

    # 2. Outliers
    if p["outliers"]["enabled"]:
        train = clip_outliers(train, method=p["outliers"]["method"], factor=p["outliers"]["factor"])
        test = clip_outliers(test, method=p["outliers"]["method"], factor=p["outliers"]["factor"])

    # 3. Constant add
    add_const = p["scaling"].get("add_constant")
    if add_const is not None:
        try:
            val = float(add_const)
            train = train.add(val)
            test = test.add(val)
        except Exception:  # noqa: BLE001
            pass

    # 4. Time features
    if p["time_features"]["enabled"]:
        train = time_to_feature(train)
        test = time_to_feature(test)

    # 5. Lags
    if p["lags"]["enabled"] and p["lags"]["n_lags"] > 0:
        train = generate_lags(train, n_lags=p["lags"]["n_lags"], step=p["lags"]["step"])
        test = generate_lags(test, n_lags=p["lags"]["n_lags"], step=p["lags"]["step"])

    # 6. Scaling
    scaler_name = p["scaling"]["method"]
    if scaler_name:
        train, fitted = scale_data(train, scale=scaler_name)
        test, _ = scale_data(test, scale=fitted)

    return train, test


def to_parquet_bytes(df: pd.DataFrame, metadata: Optional[Dict[str, str]] = None) -> bytes:
    buf = io.BytesIO()
    table = pa.Table.from_pandas(df)
    if metadata:
        # Merge existing metadata if any
        md = table.schema.metadata or {}
        merged = {**{k.encode(): v.encode() for k, v in metadata.items()}, **md}
        table = table.replace_schema_metadata(merged)
    pq.write_table(table, buf)
    return buf.getvalue()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=5), reraise=True)
def _post_with_retry(gateway: str, bucket: str, obj: str, data: bytes) -> None:
    post_file(gateway, bucket, obj, data)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=5), reraise=True)
def _get_with_retry(gateway: str, bucket: str, obj: str):
    return get_file(gateway, bucket, obj)


def _read_meta_if_exists(gateway: str, bucket: str, meta_obj: str) -> Optional[Dict[str, Any]]:
    try:
        b = _get_with_retry(gateway, bucket, meta_obj)
        if not b:
            return None
        return json.loads(b.read().decode())  # type: ignore
    except Exception:
        return None


def _write_meta(gateway: str, bucket: str, meta_obj: str, meta: Dict[str, Any]) -> None:
    _post_with_retry(gateway, bucket, meta_obj, json.dumps(meta, separators=(",", ":")).encode())


def _log(event: str, **extra):
    base = {
        "service": "preprocess",
        "event": event,
    }
    base.update({k: v for k, v in extra.items() if v is not None})
    print(json.dumps(base))


app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    gateway = os.environ.get("GATEWAY_URL", "http://fastapi-app:8000")
    # Minimal readiness: try listing (download) one configured input file
    input_bucket = os.environ.get("INPUT_BUCKET", "dataset")
    train_file = os.environ.get("TRAIN_FILE", "PobleSec.csv")
    try:
        resp = requests.get(f"{gateway}/download/{input_bucket}/{train_file}", timeout=5)
        if resp.status_code != 200:
            return JSONResponse(status_code=503, content={"status": "degraded"})
        return {"status": "ready"}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


def run_preprocess() -> None:
    start = time.time()
    identifier = os.environ.get("IDENTIFIER", "")
    gateway = os.environ.get("GATEWAY_URL", "http://fastapi-app:8000")
    topic_train = os.environ.get("PRODUCER_TOPIC_0", "training-data")
    topic_infer = os.environ.get("PRODUCER_TOPIC_1", "inference-data")
    input_bucket = os.environ.get("INPUT_BUCKET", "dataset")

    # --- Dynamic dataset selection ---
    dataset_name = os.environ.get("DATASET_NAME")  # if provided, derive train/test names unless explicitly overridden
    train_file_env = os.environ.get("TRAIN_FILE")
    test_file_env = os.environ.get("TEST_FILE")
    if dataset_name:
        train_file = train_file_env or f"{dataset_name}.csv"
        test_file = test_file_env or f"{dataset_name}_test.csv"
    else:
        train_file = train_file_env or "PobleSec.csv"
        test_file = test_file_env or "PobleSec_test.csv"

    out_bucket = os.environ.get("OUTPUT_BUCKET", "processed-data")
    out_train_base = os.environ.get("OUTPUT_TRAIN", "processed_data")
    out_test_base = os.environ.get("OUTPUT_TEST", "test_processed_data")

    # --- Subsetting / sampling controls ---
    sample_train_rows = int(os.environ.get("SAMPLE_TRAIN_ROWS", "0") or 0)
    sample_test_rows = int(os.environ.get("SAMPLE_TEST_ROWS", "0") or 0)
    sample_strategy = os.environ.get("SAMPLE_STRATEGY", "head").lower()  # head | random
    sample_seed = int(os.environ.get("SAMPLE_SEED", "42") or 42)

    active_cfg = build_active_config()
    # Embed data & sampling parameters into config for hashing/idempotency lineage
    active_cfg["_data"] = {
        "train_file": train_file,
        "test_file": test_file,
        "sample_train_rows": sample_train_rows,
        "sample_test_rows": sample_test_rows,
        "sample_strategy": sample_strategy,
    }
    canonical, config_hash = canonical_config_blob(active_cfg)
    train_obj = f"{out_train_base}.parquet"
    test_obj = f"{out_test_base}.parquet"
    train_meta_obj = f"{out_train_base}.meta.json"
    test_meta_obj = f"{out_test_base}.meta.json"

    producer = None
    try:
        producer = create_producer()
    except Exception as e:  # noqa: BLE001
        _log("producer_init_fail", error=str(e))

    # Idempotency check (train_meta sufficient)
    force_reprocess = os.environ.get("FORCE_REPROCESS", "0").lower() in {"1","true","yes"}
    existing_meta = _read_meta_if_exists(gateway, out_bucket, train_meta_obj)
    if not force_reprocess and existing_meta and existing_meta.get("config_hash") == config_hash:
        # Re-emit claim checks, skip processing
        _log("skip_idempotent", identifier=identifier, config_hash=config_hash, object_key=train_obj, result="cached")
        if producer:
            produce_message(
                producer,
                topic_train,
                {"bucket": out_bucket, "object": train_obj, "size": existing_meta.get("row_count", 0), "v": 1, "config_hash": config_hash, "identifier": identifier},
                key="train-claim",
            )
            produce_message(
                producer,
                topic_infer,
                {"bucket": out_bucket, "object": test_obj, "object_key": test_obj, "size": existing_meta.get("row_count", 0), "operation": "post: test data", "v": 1, "config_hash": config_hash, "identifier": identifier},
                key="inference-claim",
            )
        return
    if force_reprocess:
        _log("force_reprocess", identifier=identifier, reason="FORCE_REPROCESS flag set", prev_config_hash=existing_meta.get("config_hash") if existing_meta else None)

    try:
        _log("download_start", identifier=identifier, config_hash=config_hash, train_file=train_file, test_file=test_file)
        df_train = read_data(_get_with_retry(gateway, input_bucket, train_file))
        df_test = read_data(_get_with_retry(gateway, input_bucket, test_file))
        orig_train_rows, orig_test_rows = len(df_train), len(df_test)

        # Apply sampling for faster iterations if requested
        if sample_train_rows > 0:
            n = min(sample_train_rows, len(df_train))
            if sample_strategy == "random":
                df_train = df_train.sample(n=n, random_state=sample_seed)
            else:
                df_train = df_train.head(n)
        if sample_test_rows > 0:
            n2 = min(sample_test_rows, len(df_test))
            if sample_strategy == "random":
                df_test = df_test.sample(n=n2, random_state=sample_seed)
            else:
                df_test = df_test.head(n2)
        if sample_train_rows > 0 or sample_test_rows > 0:
            _log(
                "sampling_applied",
                identifier=identifier,
                config_hash=config_hash,
                train_rows_before=orig_train_rows,
                test_rows_before=orig_test_rows,
                train_rows_after=len(df_train),
                test_rows_after=len(df_test),
                strategy=sample_strategy,
            )
        _log("download_done", identifier=identifier, config_hash=config_hash, train_rows=len(df_train), test_rows=len(df_test))

        proc_train, proc_test = apply_pipeline(df_train, df_test, active_cfg)
        _log("pipeline_done", identifier=identifier, config_hash=config_hash, train_shape=list(proc_train.shape), test_shape=list(proc_test.shape))

        # Parquet metadata embed
        meta_embed = {"preprocess_config": canonical, "config_hash": config_hash}
        train_bytes = to_parquet_bytes(proc_train, meta_embed)
        test_bytes = to_parquet_bytes(proc_test, meta_embed)

        _log("upload_start", identifier=identifier, config_hash=config_hash, object_key=train_obj)
        _post_with_retry(gateway, out_bucket, train_obj, train_bytes)
        _post_with_retry(gateway, out_bucket, test_obj, test_bytes)
        _log("upload_done", identifier=identifier, config_hash=config_hash, object_key=train_obj, size_train=len(train_bytes), size_test=len(test_bytes))

        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        meta_common = {
            "identifier": identifier,
            "source_bucket": input_bucket,
            "train_source_object": train_file,
            "test_source_object": test_file,
            "config": canonical,
            "config_hash": config_hash,
            "created_at": created_at,
            "scaler_type": active_cfg["scaling"]["method"],
        }
        train_meta = {
            **meta_common,
            "output_object": train_obj,
            "row_count": len(proc_train),
            "column_names": proc_train.columns.tolist(),
        }
        test_meta = {
            **meta_common,
            "output_object": test_obj,
            "row_count": len(proc_test),
            "column_names": proc_test.columns.tolist(),
        }
        _write_meta(gateway, out_bucket, train_meta_obj, train_meta)
        _write_meta(gateway, out_bucket, test_meta_obj, test_meta)

        if producer:
            produce_message(
                producer,
                topic_train,
                {"bucket": out_bucket, "object": train_obj, "size": len(train_bytes), "v": 1, "config_hash": config_hash, "identifier": identifier},
                key="train-claim",
            )
            produce_message(
                producer,
                topic_infer,
                {"bucket": out_bucket, "object": test_obj, "object_key": test_obj, "size": len(test_bytes), "operation": "post: test data", "v": 1, "config_hash": config_hash, "identifier": identifier},
                key="inference-claim",
            )

        duration_ms = int((time.time() - start) * 1000)
        _log("success", identifier=identifier, config_hash=config_hash, object_key=train_obj, duration_ms=duration_ms, result="ok")
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.time() - start) * 1000)
        _log("failure", identifier=identifier, config_hash=config_hash, error=str(exc), duration_ms=duration_ms, result="error")
        try:
            if producer:
                publish_error(
                    producer,
                    dlq_topic="DLQ-preprocess",
                    operation="preprocess",
                    status="Failure",
                    error_details=str(exc),
                    payload={"config_hash": config_hash},
                )
        except Exception:
            pass
        raise


def main():  # pragma: no cover
    run_preprocess()


if __name__ == "__main__":  # pragma: no cover
    main()
