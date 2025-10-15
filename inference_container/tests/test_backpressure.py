import os
import time
import queue
import threading
import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

# Minimal shim to import main components without starting runtime
os.environ.setdefault("INFERENCE_AUTOSTART", "0")
os.environ.setdefault("USE_BOUNDED_QUEUE", "1")
os.environ.setdefault("QUEUE_MAXSIZE", "32")
os.environ.setdefault("ENABLE_MICROBATCH", "1")
os.environ.setdefault("BATCH_SIZE", "8")
os.environ.setdefault("BATCH_TIMEOUT_MS", "5")

# Required envs for import
os.environ.setdefault("GATEWAY_URL", "http://localhost:8000")
os.environ.setdefault("CONSUMER_TOPIC_0", "inference-data")
os.environ.setdefault("CONSUMER_TOPIC_1", "model-training")
os.environ.setdefault("PROMOTION_TOPIC", "model-selected")
os.environ.setdefault("CONSUMER_GROUP_ID", "test-group")
os.environ.setdefault("PRODUCER_TOPIC", "performance-eval")
os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

import importlib, sys, types, os
sys.path.insert(0, os.path.join(os.getcwd(), 'inference_container'))

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
        return df
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


def test_bounded_queue_and_backpressure():
    q = main.message_queue
    _drain(q)
    # Simulate enqueue via callback factory
    cb = main._kafka_callback_factory(main.inferencer, "preprocessing", q)

    # Enqueue more than QUEUE_MAXSIZE
    total = int(os.getenv("QUEUE_MAXSIZE")) * 3
    dropped = 0
    for i in range(total):
        try:
            cb(DummyMessage(key=f"k{i}", value={"bucket":"b","object":"o"}))
        except queue.Full:
            dropped += 1

    # Queue should not exceed maxsize
    assert q.qsize() <= int(os.getenv("QUEUE_MAXSIZE"))


def test_ttl_expiry():
    os.environ["ENABLE_TTL"] = "1"
    q = main.message_queue
    _drain(q)
    cb = main._kafka_callback_factory(main.inferencer, "preprocessing", q)

    expired = int(time.time()*1000) - 100
    cb(DummyMessage(key="expired", value={"bucket":"b","object":"o"}, headers=[(b'deadline_ms', str(expired).encode())]))

    # Start worker for a brief moment to drain
    t = threading.Thread(target=main.message_handler, args=(main.inferencer, q), daemon=True)
    t.start()
    time.sleep(0.05)
    assert q.qsize() == 0


def test_microbatch_drain():
    os.environ["ENABLE_MICROBATCH"] = "1"
    os.environ["BATCH_SIZE"] = "4"
    os.environ["BATCH_TIMEOUT_MS"] = "10"

    q = main.message_queue
    _drain(q)
    # Use training messages with incomplete details so worker won't try data fetch/model load
    cb = main._kafka_callback_factory(main.inferencer, "training", q)
    for i in range(7):
        cb(DummyMessage(key=f"k{i}", value={"operation":"","status":""}))

    t = threading.Thread(target=main.message_handler, args=(main.inferencer, q), daemon=True)
    t.start()
    time.sleep(0.1)
    assert q.qsize() == 0


@pytest.mark.asyncio
async def test_busy_wait_allows_pending_inference(monkeypatch):
    from inference_container import api_server

    class _DummyInferencer:
        def __init__(self):
            self.busy = True
            self.current_model = object()
            self.df = None
            self.perform_calls = 0
            self.current_run_id = "run123"
            self.model_type = "GRU"
            self.model_class = "pytorch"
            self.last_prediction_response = {}

        async def simulate_delay_if_enabled(self):  # pragma: no cover - simple stub
            return

        def perform_inference(self, df, inference_length=1):
            self.perform_calls += 1
            self.busy = True
            result_index = pd.to_datetime(["2020-01-01T00:00:00Z"])
            result = pd.DataFrame({"value": [42.0]}, index=result_index)
            self.busy = False
            return result

    dummy_inf = _DummyInferencer()
    api_server.BUSY_WAIT_TIMEOUT_MS = 200
    api_server.BUSY_WAIT_INTERVAL_MS = 5
    monkeypatch.setattr(api_server, "_get_inferencer", lambda: dummy_inf)

    loop = asyncio.get_running_loop()
    loop.call_later(0.01, setattr, dummy_inf, "busy", False)

    request = api_server.PredictRequest(
        inference_length=1,
        index_col="time",
        data={
            "time": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:01:00Z",
                "2020-01-01T00:02:00Z",
            ],
            "value": [1.0, 2.0, 3.0],
        },
    )

    try:
        response = await api_server._execute_inference(request, None, "busy-wait-test")
        assert response["status"] == "SUCCESS"
        assert dummy_inf.perform_calls == 1
    finally:
        # Drain any pending callbacks to keep the loop clean
        await asyncio.sleep(0)
