# preprocess_container/main.py — config-driven preprocessing
import os
import io
import json
from typing import Dict, Any

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

def load_config() -> Dict[str, Any]:
    config_path = os.environ.get("CONFIG_PATH", "/app/config.json")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {str(e)}")
        raise

def main():
    try:
        # Load configuration
        config = load_config()
        data_config = config["data"]
        preproc_config = config["preprocessing"]

def main():
    try:
        # Load configuration
        config = load_config()
        data_config = config["data"]
        preproc_config = config["preprocessing"]

        # Service URLs and topics
        FASTAPI_URL = os.environ.get("GATEWAY_URL", "http://fastapi-app:8000")
        TRAIN_TOPIC = os.environ.get("PRODUCER_TOPIC_0", "training-data")
        TEST_TOPIC = os.environ.get("PRODUCER_TOPIC_1", "inference-data")

        # Get training data
        print(f"Attempting to get file content from: {FASTAPI_URL}/download/{data_config['input_bucket']}/{data_config['train_file']}")
        train_df = read_data(get_file(FASTAPI_URL, data_config['input_bucket'], data_config['train_file']))
        print(f"Total size of content: {len(train_df.to_csv().encode())} bytes")

        # Get test data
        print(f"Attempting to get file content from: {FASTAPI_URL}/download/{data_config['input_bucket']}/{data_config['test_file']}")
        test_df = read_data(get_file(FASTAPI_URL, data_config['input_bucket'], data_config['test_file']))
        print(f"Total size of content: {len(test_df.to_csv().encode())} bytes")

        # Apply preprocessing steps based on configuration
        train_processed = train_df.copy()
        test_processed = test_df.copy()

        # Handle NaNs
        if preproc_config["handle_nans"]["enabled"]:
            train_processed = handle_nans(
                train_processed,
                threshold=preproc_config["handle_nans"]["threshold"],
                window=preproc_config["handle_nans"]["knn_neighbors"],
                no_drop=not preproc_config["handle_nans"]["drop_rows"]
            )
            test_processed = handle_nans(
                test_processed,
                threshold=preproc_config["handle_nans"]["threshold"],
                window=preproc_config["handle_nans"]["knn_neighbors"],
                no_drop=not preproc_config["handle_nans"]["drop_rows"]
            )

        # Clip outliers
        if preproc_config["outliers"]["enabled"]:
            train_processed = clip_outliers(
                train_processed,
                method=preproc_config["outliers"]["method"],
                factor=preproc_config["outliers"]["factor"]
            )
            test_processed = clip_outliers(
                test_processed,
                method=preproc_config["outliers"]["method"],
                factor=preproc_config["outliers"]["factor"]
            )

        # Add constant (if specified)
        if preproc_config["scaling"].get("add_constant"):
            try:
                val = float(preproc_config["scaling"]["add_constant"])
                train_processed = train_processed.add(val)
                test_processed = test_processed.add(val)
            except (ValueError, TypeError):
                pass

        # Time features
        if preproc_config["time_features"]["enabled"]:
            train_processed = time_to_feature(train_processed)
            test_processed = time_to_feature(test_processed)

        # Generate lags
        if preproc_config["lags"]["enabled"] and preproc_config["lags"]["n_lags"] > 0:
            train_processed = generate_lags(
                train_processed,
                n=preproc_config["lags"]["n_lags"],
                step=preproc_config["lags"]["step"]
            )
            test_processed = generate_lags(
                test_processed,
                n=preproc_config["lags"]["n_lags"],
                step=preproc_config["lags"]["step"]
            )

        # Scale data
        scaler = None
        if preproc_config["scaling"]["method"]:
            train_processed, scaler = scale_data(
                train_processed,
                scaler_name=preproc_config["scaling"]["method"]
            )
            test_processed, _ = scale_data(
                test_processed,
                fitted_scaler=scaler
            )

        # Save processed data
        train_table = pa.Table.from_pandas(train_processed)
        test_table = pa.Table.from_pandas(test_processed)

        # Convert to parquet
        train_buffer = io.BytesIO()
        test_buffer = io.BytesIO()
        pq.write_table(train_table, train_buffer)
        pq.write_table(test_table, test_buffer)

        # Upload processed files
        train_size = len(train_buffer.getvalue())
        print(f"Preparing to upload {train_size} bytes of data.")
        post_file(
            FASTAPI_URL,
            data_config['output_bucket'],
            f"{data_config['output_train']}.parquet",
            train_buffer.getvalue(),
            content_type='application/octet-stream'
        )

        test_size = len(test_buffer.getvalue())
        print(f"Preparing to upload {test_size} bytes of data.")
        post_file(
            FASTAPI_URL,
            data_config['output_bucket'],
            f"{data_config['output_test']}.parquet",
            test_buffer.getvalue(),
            content_type='application/octet-stream'
        )

        # Send Kafka messages
        producer = create_producer()
        
        train_message = {
            "bucket": data_config['output_bucket'],
            "object": f"{data_config['output_train']}.parquet",
            "size": train_size
        }
        produce_message(producer, TRAIN_TOPIC, "train-claim", train_message)
        print(f"Successfully sent JSON message with key 'train-claim' to topic '{TRAIN_TOPIC}'.")

        test_message = {
            "bucket": data_config['output_bucket'],
            "object": f"{data_config['output_test']}.parquet",
            "size": test_size
        }
        produce_message(producer, TEST_TOPIC, "inference-claim", test_message)
        print(f"Successfully sent JSON message with key 'inference-claim' to topic '{TEST_TOPIC}'.")

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        publish_error("preprocess", str(e))
        raise

if __name__ == "__main__":
    main()

def to_parquet_bytes(df: pd.DataFrame, *, scaler_obj: Any, config: Dict[str, Any]) -> bytes:
    table = pa.Table.from_pandas(df, preserve_index=True)
    meta = dict(table.schema.metadata or {})
    try:
        ser = pickle.dumps(scaler_obj) if scaler_obj is not None else b""
    except Exception:
        ser = b""
    meta[b"scaler_object"] = ser
    meta[b"scaler_type"] = (type(scaler_obj).__name__ if scaler_obj is not None else "None").encode("utf-8")
    meta[b"preprocess_config"] = json.dumps(config, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    table = table.replace_schema_metadata(meta)
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="snappy")
    return sink.getvalue()

def main():
    producer = create_producer()
    druid = DruidIngester() if DRUID_ENABLE else None

    raw_bytes  = get_file(FASTAPI_URL, INPUT_BUCKET, OBJECT_NAME)
    if raw_bytes is None:
        raise RuntimeError(f"Failed to download {INPUT_BUCKET}/{OBJECT_NAME} from {FASTAPI_URL}")

    test_bytes = get_file(FASTAPI_URL, INPUT_BUCKET, TEST_OBJECT_NAME) or raw_bytes

    df_raw  = read_data(raw_bytes)
    df_test = read_data(test_bytes)

    # Optional filter by identifier if a known column exists
    if IDENTIFIER and isinstance(df_raw, pd.DataFrame):
        for col in ("District","district","identifier","Identifier","neighborhood","Neighborhood"):
            if col in df_raw.columns: df_raw = df_raw[df_raw[col] == IDENTIFIER]
    if IDENTIFIER and isinstance(df_test, pd.DataFrame):
        for col in ("District","district","identifier","Identifier","neighborhood","Neighborhood"):
            if col in df_test.columns: df_test = df_test[df_test[col] == IDENTIFIER]

    try:
        df_proc, scaler = apply_pipeline(df_raw)
        df_test_proc, _ = apply_pipeline(df_test, fitted_scaler=scaler)
    except Exception as e:
        publish_error(
            producer, dlq_topic=f"DLQ-{TRAIN_TOPIC}",
            operation="Preprocess", status="Failure",
            error_details=f"Preprocessing failed: {e}\n{traceback.format_exc()}",
            payload={"bucket": INPUT_BUCKET, "object": OBJECT_NAME, "identifier": IDENTIFIER},
        )
        raise

    if druid is not None:
        try:
            druid_df = df_test_proc.reset_index(names=DRUID_TIME_COLUMN)
            task_id = druid.ingest_dataframe(druid_df, DRUID_DATASOURCE, DRUID_TIME_COLUMN)
            print(f"Druid ingestion task id: {task_id}")
        except Exception as e:
            print(f"Druid ingestion failed: {e}")

    config = {
        "identifier": IDENTIFIER,
        "handle_nans": {"enabled": ENABLE_HANDLE_NANS, "threshold": NANS_THRESHOLD, "knn": NANS_KNN, "drop_rows": NANS_DROP_ROWS},
        "clip": {"enabled": ENABLE_CLIP, "method": CLIP_METHOD, "factor": CLIP_FACTOR},
        "time_features": {"enabled": ENABLE_TIME_FEATURES},
        "lags": {"enabled": ENABLE_LAGS, "n": LAGS_N, "step": LAGS_STEP},
        "scaler": SCALER or "",
        "add_val": ADD_VAL or "",
    }

    # Upload train parquet
    try:
        train_bytes = to_parquet_bytes(df_proc, scaler_obj=scaler, config=config)
        ok = post_file(FASTAPI_URL, OUTPUT_BUCKET, f"{OUTPUT_FILENAME}.parquet", train_bytes)
        if not ok: raise RuntimeError("Upload returned non-OK")
    except Exception as e:
        publish_error(
            producer, dlq_topic=f"DLQ-{TRAIN_TOPIC}",
            operation="Post File to MinIO", status="Failure",
            error_details=f"Failed to post processed data: {e}\n{traceback.format_exc()}",
            payload={"endpoint": FASTAPI_URL, "bucket": OUTPUT_BUCKET, "object": f"{OUTPUT_FILENAME}.parquet"},
        )
        raise

    # Upload test parquet
    try:
        test_bytes_out = to_parquet_bytes(df_test_proc, scaler_obj=scaler, config=config)
        ok2 = post_file(FASTAPI_URL, OUTPUT_BUCKET, f"{OUTPUT_TEST_FILENAME}.parquet", test_bytes_out)
        if not ok2: raise RuntimeError("Upload test returned non-OK")
    except Exception as e:
        publish_error(
            producer, dlq_topic=f"DLQ-{TEST_TOPIC}",
            operation="Post File to MinIO", status="Failure",
            error_details=f"Failed to post test data: {e}\n{traceback.format_exc()}",
            payload={"endpoint": FASTAPI_URL, "bucket": OUTPUT_BUCKET, "object": f"{OUTPUT_TEST_FILENAME}.parquet"},
        )
        raise

    # Claim-check messages
    produce_message(
        producer, topic=TRAIN_TOPIC, key="train-claim",
        value={"operation": "post: train data", "status": "SUCCESS",
               "bucket": OUTPUT_BUCKET, "object_key": f"{OUTPUT_FILENAME}.parquet",
               "identifier": IDENTIFIER}
    )
    produce_message(
        producer, topic=TEST_TOPIC, key="inference-claim",
        value={"operation": "post: inference data", "status": "SUCCESS",
               "bucket": OUTPUT_BUCKET, "object_key": f"{OUTPUT_TEST_FILENAME}.parquet",
               "identifier": IDENTIFIER}
    )

if __name__ == "__main__":
    main()
