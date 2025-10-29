import pytest

pytest.skip("Legacy queue-based tests retained for history; see test_sync_predict.py for current coverage.", allow_module_level=True)

import asyncio
import concurrent.futures
import os
import queue
import threading
import time

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from inference_container import api_server

client = TestClient(api_server.app)

TIME_FEATURE_COLUMNS = {
    "min_of_day_sin",
    "min_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
}

import importlib, sys, types, os
_PROJECT_ROOT = os.getcwd()
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'inference_container'))
sys.path.insert(0, _PROJECT_ROOT)

# Mock heavy modules and kafka before importing main
for mod in [ 'pyarrow', 'pyarrow.parquet', 'mlflow' ]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

# Minimal torch stub with device and cuda.is_available
if 'torch' not in sys.modules:
    tmod = types.ModuleType('torch')
    class _Cuda:
        @staticmethod
        def is_available():
            return False
    def device(name):
        return name
    tmod.cuda = _Cuda()
    tmod.device = device
    sys.modules['torch'] = tmod

# Provide a minimal pandas stub only if the real package is unavailable
if 'pandas' not in sys.modules:
    pdmod = types.ModuleType('pandas')
    class Series: pass
    class DataFrame: pass
    class Timedelta:
        def __init__(self, minutes=1, **kwargs):
            self._minutes = minutes
    pdmod.Series = Series
    pdmod.DataFrame = DataFrame
    pdmod.Timedelta = Timedelta
    sys.modules['pandas'] = pdmod

# Add mlflow submodules used by code
if 'mlflow' in sys.modules:
    if 'mlflow.artifacts' not in sys.modules:
        mlfa = types.ModuleType('mlflow.artifacts')
        def download_artifacts(artifact_uri, dst_path=None):
            return ''
        mlfa.download_artifacts = download_artifacts
        sys.modules['mlflow.artifacts'] = mlfa
    if 'mlflow.pyfunc' not in sys.modules:
        mlfp = types.ModuleType('mlflow.pyfunc')
        class DummyModel:
            def predict(self, x):
                import numpy as np
                # output one-step target
                return np.zeros((1,1,1))
        def load_model(uri):
            return DummyModel()
        def get_model_dependencies(uri):
            return []
        mlfp.load_model = load_model
        mlfp.get_model_dependencies = get_model_dependencies
        sys.modules['mlflow.pyfunc'] = mlfp

# Minimal kafka shim for import-time producer/consumer
if 'kafka' not in sys.modules:
    kafka_mod = types.ModuleType('kafka')
    class DummyProducer:
        def __init__(self, *a, **k):
            pass
        def send(self, *a, **k):
            return None
        def flush(self):
            return None
        def close(self):
            return None
    class DummyConsumer:
        def __init__(self, *a, **k):
            self._assigned = []
        def assignment(self):
            return self._assigned
        def pause(self, *parts):
            self._assigned = list(parts)
        def resume(self, *parts):
            return None
        def poll(self, *a, **k):
            return {}
        def close(self):
            return None
        def commit(self, *a, **k):
            return None
    kafka_mod.KafkaProducer = DummyProducer
    kafka_mod.KafkaConsumer = DummyConsumer
    # structs submodule
    structs = types.ModuleType('kafka.structs')
    class TopicPartition:
        def __init__(self, topic, partition):
            self.topic = topic
            self.partition = partition
    class OffsetAndMetadata:
        def __init__(self, offset, metadata):
            self.offset = offset
            self.metadata = metadata
    structs.TopicPartition = TopicPartition
    structs.OffsetAndMetadata = OffsetAndMetadata
    sys.modules['kafka'] = kafka_mod
    sys.modules['kafka.structs'] = structs

# Stub data_utils used by inferencer
if 'data_utils' not in sys.modules:
    du = types.ModuleType('data_utils')
    def window_data(df, *args, **kwargs):
        return [], [], None
    def check_uniform(df):
        import pandas as pd
        return pd.Timedelta(minutes=1)
    def time_to_feature(df):
        df_copy = df.copy()
        feature_cols = [
            "min_of_day_sin",
            "min_of_day_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "day_of_year_sin",
            "day_of_year_cos",
        ]
        for col in feature_cols:
            if col not in df_copy.columns:
                df_copy[col] = 0.0
        return df_copy
    def subset_scaler(s, orig_cols, cols):
        return s
    def strip_timezones(df):
        return df, {"index": False, "columns": []}
    du.window_data = window_data
    du.check_uniform = check_uniform
    du.time_to_feature = time_to_feature
    du.subset_scaler = subset_scaler
    du.strip_timezones = strip_timezones
    sys.modules['data_utils'] = du

# Stub sklearn preprocessing used by inferencer
if 'sklearn' not in sys.modules:
    skl = types.ModuleType('sklearn')
    sys.modules['sklearn'] = skl
if 'sklearn.preprocessing' not in sys.modules:
    sklp = types.ModuleType('sklearn.preprocessing')
    class _Scaler:
        def inverse_transform(self, X):
            return X
    sklp.MinMaxScaler = _Scaler
    sklp.StandardScaler = _Scaler
    sklp.RobustScaler = _Scaler
    sklp.MaxAbsScaler = _Scaler
    sys.modules['sklearn.preprocessing'] = sklp

main = importlib.import_module('inference_container.main')

class DummyMessage:
    def __init__(self, key: str, value: dict, headers=None):
        self.key = key
        self.value = value
        self.headers = headers or []

def _drain(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        return


def _reset_queue_metrics():
    from inference_container import api_server

    for key in list(api_server.queue_metrics.keys()):
        if key.endswith("_type"):
            api_server.queue_metrics[key] = None
        else:
            api_server.queue_metrics[key] = 0


def _build_prepared_dataframe(data_columns=None):
    data_columns = data_columns or {
        "ts": ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
        "value": [1.0, 2.0, 3.0],
        "up": [10.0, 11.0, 12.0],
    }
    df = pd.DataFrame(data_columns).copy()
    if "ts" not in df.columns:
        raise AssertionError("test helper requires 'ts' column")
    timestamps = pd.to_datetime(df.pop("ts"))
    df.index = timestamps
    df = df.sort_index()
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    base_columns = [col for col in df.columns if col not in TIME_FEATURE_COLUMNS]
    for feature in TIME_FEATURE_COLUMNS:
        if feature not in df.columns:
            df[feature] = 0.0
    return df, base_columns


def _make_process_pool_payload(data_columns=None, inference_length=2, req_id="req123"):
    from inference_container import process_pool

    df, base_columns = _build_prepared_dataframe(data_columns)
    return process_pool.build_job_payload(df, inference_length, req_id, {}, expected_base_columns=base_columns)


def test_run_inference_job_consumes_prepared_dataframe(monkeypatch):
    from inference_container import process_pool

    df_prepared, base_columns = _build_prepared_dataframe({
        "ts": [
            "2024-01-01T00:00:00",
            "2024-01-01T00:01:00",
            "2024-01-01T00:02:00",
            "2024-01-01T00:03:00",
        ],
        "value": [1.0, 2.0, 3.0, 4.0],
        "up": [10.0, 11.0, 12.0, 13.0],
    })

    class _DummyInferencer:
        def __init__(self):
            self.current_model = object()
            self.current_run_id = "run123"
            self.current_config_hash = "hash"
            self.model_type = "GRU"
            self.last_prediction_response = None

        def perform_inference(self, df, inference_length=None):
            pd.testing.assert_frame_equal(df, df_prepared)
            rows = max(1, inference_length or 1)
            idx = pd.date_range("2024-01-01", periods=rows, freq="1min")
            return pd.DataFrame({"value": list(range(rows))}, index=idx)

    dummy = _DummyInferencer()
    monkeypatch.setattr(process_pool, "_ensure_worker_inferencer", lambda snapshot: dummy)

    payload = process_pool.build_job_payload(df_prepared.copy(), 3, "req123", {}, expected_base_columns=base_columns)
    result = process_pool._run_inference_job(payload)

    assert isinstance(result, dict)
    assert result["response"]["status"] == "SUCCESS"
    assert len(result["response"]["predictions"]) == 3
    assert result["meta"]["worker_id"] == os.getpid()
    assert result["meta"]["predictions"] == 3


def test_run_inference_job_missing_columns_returns_400(monkeypatch):
    from inference_container import process_pool

    class _ErrorInferencer:
        def __init__(self):
            self.current_model = object()
            self.current_run_id = "run123"
            self.current_config_hash = "hash"
            self.model_type = "GRU"

        def perform_inference(self, df, inference_length=None):
            raise KeyError("['value'] not found in axis")

    dummy = _ErrorInferencer()
    monkeypatch.setattr(process_pool, "_ensure_worker_inferencer", lambda snapshot: dummy)

    df_prepared, base_columns = _build_prepared_dataframe()
    df_missing_value = df_prepared.drop(columns=["value"])
    payload = process_pool.build_job_payload(
        df_missing_value,
        1,
        "req123",
        {},
        expected_base_columns=base_columns,
    )

    with pytest.raises(process_pool.InferenceHTTPError) as excinfo:
        process_pool._run_inference_job(payload)

    assert excinfo.value.status_code == 400
    assert "Missing required columns for inference" in excinfo.value.detail
    assert excinfo.value.worker_id == os.getpid()


@pytest.mark.asyncio
async def test_predict_invalid_payload_returns_400_without_submitting(monkeypatch):
    from inference_container import api_server

    _reset_queue_metrics()
    monkeypatch.setattr(api_server, "_cache_enabled", lambda: False)

    ensure_calls = {"count": 0}
    snapshot_calls = {"count": 0}
    slot_calls = {"count": 0}
    submit_calls: list = []

    def _ensure(snapshot):
        ensure_calls["count"] += 1

    def _snapshot():
        snapshot_calls["count"] += 1
        return {"run_id": "run123"}

    def _try_acquire():
        slot_calls["count"] += 1
        return True

    class _StubInferencer:
        def __init__(self):
            idx = pd.date_range("2024-01-01", periods=3, freq="1min")
            self.df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "up": [10.0, 11.0, 12.0]}, index=idx)
            self.current_model = object()
            self.current_run_id = "run123"
            self.model_type = "GRU"
            self.current_config_hash = "hash"
            self.last_prediction_response = {}

    stub_service = _StubInferencer()

    monkeypatch.setattr(api_server, "_ensure_process_pool_ready", _ensure)
    monkeypatch.setattr(api_server, "_current_model_snapshot", _snapshot)
    monkeypatch.setattr(api_server, "try_acquire_slot", _try_acquire)
    monkeypatch.setattr(api_server, "release_slot", lambda: None)
    monkeypatch.setattr(api_server, "pending_jobs", lambda: 0)
    monkeypatch.setattr(api_server, "_get_inferencer", lambda: stub_service)

    result_payload = {
        "status": "SUCCESS",
        "identifier": "default",
        "run_id": "run123",
        "predictions": [{"ts": "2024-01-01T00:03:00", "value": 3.0}],
    }

    def _submit(payload):
        submit_calls.append(payload)
        assert isinstance(payload["prepared_df"], pd.DataFrame)
        for feature in TIME_FEATURE_COLUMNS:
            assert feature in payload["prepared_df"].columns
        future = concurrent.futures.Future()
        future.set_result(result_payload)
        return future

    monkeypatch.setattr(api_server, "submit_inference_job", _submit)

    invalid_request = api_server.PredictRequest(
        data={"value": [1.0, 2.0, 3.0], "up": [10.0, 11.0, 12.0]},
    )

    with pytest.raises(HTTPException) as excinfo:
        await api_server.predict(invalid_request, inference_length=2)

    assert excinfo.value.status_code == 400
    assert "timestamp" in excinfo.value.detail
    assert ensure_calls["count"] == 0
    assert snapshot_calls["count"] == 0
    assert slot_calls["count"] == 0
    assert submit_calls == []
    assert api_server.queue_metrics["enqueued"] == 0
    assert api_server.queue_metrics["completed"] == 0
    assert api_server.queue_metrics["active"] == 0

    valid_request = api_server.PredictRequest(
        data={
            "ts": ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            "value": [1.0, 2.0, 3.0],
            "up": [10.0, 11.0, 12.0],
        }
    )

    response = await api_server.predict(valid_request, inference_length=2)

    assert response == result_payload
    assert ensure_calls["count"] == 1
    assert snapshot_calls["count"] == 1
    assert slot_calls["count"] == 1
    assert len(submit_calls) == 1
    assert api_server.queue_metrics["enqueued"] == 1
    assert api_server.queue_metrics["completed"] == 1
    assert api_server.queue_metrics["active"] == 0


@pytest.mark.asyncio
async def test_predict_process_pool_success(monkeypatch):
    from inference_container import api_server

    _reset_queue_metrics()
    monkeypatch.setattr(api_server, "_cache_enabled", lambda: False)
    monkeypatch.setattr(api_server, "_ensure_process_pool_ready", lambda snapshot: None)
    monkeypatch.setattr(api_server, "_current_model_snapshot", lambda: {"run_id": "run123"})
    monkeypatch.setattr(api_server, "pending_jobs", lambda: 0)
    monkeypatch.setattr(api_server, "try_acquire_slot", lambda: True)
    monkeypatch.setattr(api_server, "release_slot", lambda: None)
    monkeypatch.delenv("PREDICT_FORCE_OK", raising=False)

    class _StubInferencer:
        def __init__(self):
            self.current_model = object()
            self.df = None
            self.last_prediction_response = {}
            self.current_run_id = "run123"
            self.model_type = "GRU"
            self.current_config_hash = "hash"

    stub = _StubInferencer()
    monkeypatch.setattr(api_server, "_get_inferencer", lambda: stub)

    future = concurrent.futures.Future()
    result_payload = {
        "status": "SUCCESS",
        "identifier": "default",
        "run_id": "run123",
        "predictions": [],
    }
    future.set_result(result_payload)
    monkeypatch.setattr(api_server, "submit_inference_job", lambda payload: future)

    response = await api_server.predict(api_server.PredictRequest(), inference_length=1)
    assert response == result_payload
    assert api_server.queue_metrics["completed"] == 1
    assert api_server.queue_metrics["active"] == 0


@pytest.mark.asyncio
async def test_predict_process_pool_timeout(monkeypatch):
    from inference_container import api_server

    _reset_queue_metrics()
    monkeypatch.setattr(api_server, "_cache_enabled", lambda: False)
    monkeypatch.setattr(api_server, "_ensure_process_pool_ready", lambda snapshot: None)
    monkeypatch.setattr(api_server, "_current_model_snapshot", lambda: {"run_id": "run123"})
    monkeypatch.setattr(api_server, "pending_jobs", lambda: 0)
    monkeypatch.setattr(api_server, "try_acquire_slot", lambda: True)
    monkeypatch.setattr(api_server, "release_slot", lambda: None)
    monkeypatch.setattr(api_server, "INFERENCE_TIMEOUT", 0.01, raising=False)
    monkeypatch.delenv("PREDICT_FORCE_OK", raising=False)

    class _StubInferencer:
        def __init__(self):
            self.current_model = object()
            self.df = None
            self.last_prediction_response = {}
            self.current_run_id = "run123"
            self.model_type = "GRU"
            self.current_config_hash = "hash"

    stub = _StubInferencer()
    monkeypatch.setattr(api_server, "_get_inferencer", lambda: stub)

    stalled_future = concurrent.futures.Future()
    monkeypatch.setattr(api_server, "submit_inference_job", lambda payload: stalled_future)

    with pytest.raises(HTTPException) as excinfo:
        await api_server.predict(api_server.PredictRequest(), inference_length=1)

    assert excinfo.value.status_code == 504
    assert api_server.queue_metrics["timeouts"] == 1
    assert api_server.queue_metrics["active"] == 0


@pytest.mark.asyncio
async def test_predict_queue_full_returns_429(monkeypatch):
    from inference_container import api_server

    _reset_queue_metrics()
    monkeypatch.setattr(api_server, "_cache_enabled", lambda: False)
    monkeypatch.setattr(api_server, "_ensure_process_pool_ready", lambda snapshot: None)
    monkeypatch.setattr(api_server, "_current_model_snapshot", lambda: {"run_id": "run123"})
    monkeypatch.setattr(api_server, "pending_jobs", lambda: api_server.QUEUE_MAXSIZE)
    monkeypatch.setattr(api_server, "try_acquire_slot", lambda: False)
    monkeypatch.delenv("PREDICT_FORCE_OK", raising=False)

    class _StubInferencer:
        def __init__(self):
            self.current_model = object()
            self.df = None
            self.last_prediction_response = {}
            self.current_run_id = "run123"
            self.model_type = "GRU"
            self.current_config_hash = "hash"

    stub = _StubInferencer()
    monkeypatch.setattr(api_server, "_get_inferencer", lambda: stub)

    called = False

    def _submit(_payload):
        nonlocal called
        called = True
        return concurrent.futures.Future()

    monkeypatch.setattr(api_server, "submit_inference_job", _submit)

    with pytest.raises(HTTPException) as excinfo:
        await api_server.predict(api_server.PredictRequest(), inference_length=1)

    assert excinfo.value.status_code == 429
    assert api_server.queue_metrics["rejected_full"] == 1
    assert called is False


def test_metrics_reports_synchronous_mode(monkeypatch):
    inferencer = DummyInferencer()
    monkeypatch.setattr(api_server, "_get_inferencer", lambda: inferencer)

    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "synchronous"
    assert body["workers"] == 1


class DummyInferencer:
    def __init__(self, df: pd.DataFrame | None = None, expected_columns: list[str] | None = None):
        self.df = df
        self.current_model = object()
        self.current_run_id = "run123"
        self.current_model_uri = "runs:/run123/model"
        self.current_config_hash = "cfg123"
        self.model_type = "dummy"
        self.expected_feature_columns = expected_columns or ["value", "up"]
        self.last_prediction_response = None

    async def simulate_delay_if_enabled(self):  # pragma: no cover - optional behaviour
        return None

    def perform_inference(self, df: pd.DataFrame, inference_length: int | None = None) -> pd.DataFrame:
        rows = int(inference_length or max(1, len(df)))
        index = pd.date_range("2024-01-01", periods=rows, freq="1min")
        return pd.DataFrame({"value": [float(i) for i in range(rows)]}, index=index)
