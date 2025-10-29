import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

TEST_DIR = Path(__file__).resolve().parent
PKG_ROOT = TEST_DIR.parent
sys.path.insert(0, str(PKG_ROOT))

# Provide lightweight sklearn shims when the real package is unavailable (CI environments).
if "sklearn" not in sys.modules:
    sys.modules["sklearn"] = types.ModuleType("sklearn")

if "sklearn.preprocessing" not in sys.modules:
    preprocessing = types.ModuleType("sklearn.preprocessing")

    class _Scaler:
        def fit_transform(self, data):
            return np.asarray(data)

        def transform(self, data):
            return np.asarray(data)

    preprocessing.MinMaxScaler = _Scaler
    preprocessing.StandardScaler = _Scaler
    preprocessing.RobustScaler = _Scaler
    preprocessing.MaxAbsScaler = _Scaler
    sys.modules["sklearn.preprocessing"] = preprocessing

if "sklearn.impute" not in sys.modules:
    impute = types.ModuleType("sklearn.impute")

    class KNNImputer:
        def __init__(self, *_, **__):
            return

        def fit_transform(self, data):
            return np.asarray(data)

    impute.KNNImputer = KNNImputer
    sys.modules["sklearn.impute"] = impute

from inference_container import api_server


client = TestClient(api_server.app)


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


@pytest.fixture(autouse=True)
def _baseline(monkeypatch):
    monkeypatch.setenv("ENABLE_PREDICT_CACHE", "0")
    monkeypatch.setenv("PREDICT_FORCE_OK", "0")
    for key in api_server.queue_metrics:
        api_server.queue_metrics[key] = 0 if key != "last_error_type" else None
    yield
    for key in api_server.queue_metrics:
        api_server.queue_metrics[key] = 0 if key != "last_error_type" else None


def _build_request_payload() -> dict:
    return {
        "data": {
            "ts": ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            "value": [1.0, 2.0, 3.0],
            "up": [10.0, 11.0, 12.0],
        },
        "inference_length": 3,
    }


def test_predict_returns_success_with_payload(monkeypatch):
    inferencer = DummyInferencer()
    monkeypatch.setattr(api_server, "_get_inferencer", lambda: inferencer)

    response = client.post("/predict", json=_build_request_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert body["identifier"] == "default"
    assert len(body["predictions"]) == 3
    assert api_server.queue_metrics["completed"] == 1


def test_predict_uses_cached_dataframe_when_no_payload(monkeypatch):
    cached_index = pd.date_range("2024-01-01", periods=2, freq="1min")
    cached_df = pd.DataFrame({"value": [5.0, 6.0], "up": [7.0, 8.0]}, index=cached_index)
    inferencer = DummyInferencer(df=cached_df)
    monkeypatch.setattr(api_server, "_get_inferencer", lambda: inferencer)

    response = client.post("/predict", json={})

    assert response.status_code == 200
    assert api_server.queue_metrics["enqueued"] == 1
    assert api_server.queue_metrics["completed"] == 1


def test_predict_missing_required_columns_returns_400(monkeypatch):
    inferencer = DummyInferencer()
    monkeypatch.setattr(api_server, "_get_inferencer", lambda: inferencer)

    payload = {
        "data": {
            "ts": ["2024-01-01T00:00:00"],
            "up": [1.0],
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert api_server.queue_metrics["completed"] == 0
    assert api_server.queue_metrics["error_500_total"] == 0


def test_predict_returns_503_when_inferencer_unavailable(monkeypatch):
    def _raise():
        raise RuntimeError("inferencer unavailable")

    monkeypatch.setattr(api_server, "_get_inferencer", _raise)

    response = client.post("/predict", json=_build_request_payload())

    assert response.status_code == 503
    assert api_server.queue_metrics["error_500_total"] == 1


def test_metrics_reports_synchronous_mode(monkeypatch):
    inferencer = DummyInferencer()
    monkeypatch.setattr(api_server, "_get_inferencer", lambda: inferencer)

    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "synchronous"
    assert body["workers"] == 1
