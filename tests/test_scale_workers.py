import asyncio
import time
import pytest
from fastapi.testclient import TestClient
from inference_container import api_server as api

client = TestClient(api.app)


def test_scale_workers_up_and_down(monkeypatch):
    # ensure no real model or heavy inference runs
    class DummyInf:
        current_model = None
        df = None
    monkeypatch.setattr(api, "_get_inferencer", lambda: DummyInf())

    # Ensure workers are initialized
    # call startup handler to create initial workers
    import asyncio as _asyncio
    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)
    loop.run_until_complete(api._start_workers_once())

    old_count = len(api._worker_tasks)
    assert old_count >= 1

    # Scale up
    resp = client.post("/scale_workers", json={"workers": old_count + 1})
    assert resp.status_code == 200
    j = resp.json()
    assert j["new_workers"] == old_count + 1

    # Scale down
    resp2 = client.post("/scale_workers", json={"workers": max(1, old_count)})
    assert resp2.status_code == 200
    j2 = resp2.json()
    assert j2["new_workers"] == max(1, old_count)

*** End Patch