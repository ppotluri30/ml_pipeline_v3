import requests
import time
import threading
import datetime as dt

host = "http://keda-add-ons-http-interceptor-proxy.keda.svc.cluster.local:8080"
headers = {"Host": "inference-http", "Content-Type": "application/json"}

def build_payload(seq):
    base_time = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    ts_values = [(base_time + dt.timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(30)]
    return {
        "index_col": "ts",
        "inference_length": 1,
        "data": {
            "ts": ts_values,
            "down": [5000000.0 + i*1000 for i in range(30)],
            "up": [1000.0 + i for i in range(30)],
            "rnti_count": [2000.0 + i for i in range(30)],
            "mcs_down": [10.0 + (i % 5) for i in range(30)],
            "mcs_down_var": [50.0 + i*0.5 for i in range(30)],
            "mcs_up": [12.0 + (i % 4) for i in range(30)],
            "mcs_up_var": [40.0 + i*0.4 for i in range(30)],
            "rb_down": [0.05 + i*0.001 for i in range(30)],
            "rb_down_var": [1e-7 + i*1e-9 for i in range(30)],
            "rb_up": [0.01 + i*0.0005 for i in range(30)],
            "rb_up_var": [5e-8 + i*1e-9 for i in range(30)],
        }
    }

results = {"success": 0, "fail": 0, "latencies": []}
lock = threading.Lock()

def worker(seq):
    try:
        start = time.time()
        r = requests.post(host + "/predict", headers=headers, json=build_payload(seq), timeout=30)
        latency = (time.time() - start) * 1000
        with lock:
            if r.status_code == 200:
                results["success"] += 1
            else:
                results["fail"] += 1
            results["latencies"].append(latency)
    except Exception as e:
        with lock:
            results["fail"] += 1

print("Starting 60s load test through KEDA proxy...")
start_time = time.time()
seq = 0
threads = []
while time.time() - start_time < 60:
    batch = []
    for _ in range(10):
        t = threading.Thread(target=worker, args=(seq,))
        t.start()
        batch.append(t)
        seq += 1
    time.sleep(0.1)
    threads = [t for t in threads if t.is_alive()]
    threads.extend(batch)

for t in threads:
    t.join(timeout=5)

latencies = sorted(results["latencies"])
p50 = latencies[len(latencies)//2] if latencies else 0
p95 = latencies[int(len(latencies)*0.95)] if latencies else 0
p99 = latencies[int(len(latencies)*0.99)] if latencies else 0
total = results["success"] + results["fail"]
print("Total:", total, "Success:", results["success"], "Fail:", results["fail"])
print("P50:", int(p50), "ms, P95:", int(p95), "ms, P99:", int(p99), "ms")
print("RPS:", round(total/60, 1))
