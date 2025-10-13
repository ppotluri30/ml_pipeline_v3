import time
import os
import threading
import pytest
from fastapi.testclient import TestClient

# Import the app from inference_container
from inference_container import api_server as api

client = TestClient(api.app)


def test_healthz_has_queue_and_ready(monkeypatch):
    # Prevent real model loading by mocking _get_inferencer
    class DummyInf:
        current_model = None
        df = None

    monkeypatch.setattr(api, "_get_inferencer", lambda: DummyInf())

    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert "queue_length" in data
    assert "model_ready" in data


@pytest.mark.timeout(10)
def test_prometheus_metrics_endpoint(monkeypatch):
    # Ensure prometheus is available; if not skip
    if not getattr(api, "_PROMETHEUS_AVAILABLE", False):
        pytest.skip("prometheus_client not available in test environment")

    # Start the exporter in a background thread (as startup would do)
    try:
        threading.Thread(target=lambda: api.start_http_server(9091), daemon=True).start()
    except Exception:
        # start_http_server attached to api module
        pass

    # Small sleep to allow server to start
    time.sleep(0.5)

    # Scrape the metrics endpoint via TestClient (direct call to exporter not possible),
    # but TestClient cannot reach exporter started on 9091; instead assert metric objects exist.
    # Check that Gauge/Counter objects are present in module
    assert hasattr(api, "QUEUE_LEN")
    assert hasattr(api, "JOBS_PROCESSED")

    # Set and increment metrics and ensure no exceptions
    api.QUEUE_LEN.set(3)
    api.JOBS_PROCESSED.inc()

    # Read the /metrics path via the TestClient to ensure JSON metrics endpoint still exists
    resp = client.get("/metrics")
    assert resp.status_code == 200
    j = resp.json()
    assert "queue_length" in j
    assert "completed" in j

*** End Patch