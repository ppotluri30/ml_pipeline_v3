import json
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import subprocess

with open('payload-valid.json','r') as f:
    payload = json.load(f)

url = 'http://localhost:8023/predict'
USER_COUNT = 200
TEST_DURATION = 60
WAIT_MIN = 1.0
WAIT_MAX = 2.0
METRIC_SAMPLE_INTERVAL = 5.0

start = time.time()
end_time = start + TEST_DURATION

status_counts = {}
latencies = []
errors = []
metric_samples: deque[dict[str, float]] = deque()
metrics_stop = threading.Event()


def metrics_worker() -> None:
    while not metrics_stop.is_set():
        try:
            snap = requests.get('http://localhost:8023/metrics', timeout=5).json()
            snap['ts'] = time.time()
            try:
                stats = subprocess.run(
                    ['docker', 'stats', '--no-stream', '--format', '{{.CPUPerc}}', 'flts-main-inference-1'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                cpu_text = stats.stdout.strip().rstrip('%')
                snap['cpu_pct'] = float(cpu_text) if cpu_text else None
            except Exception:
                snap['cpu_pct'] = None
            metric_samples.append(snap)
        except Exception as exc:  # pragma: no cover - diagnostics only
            errors.append(f"metrics:{exc}")
        metrics_stop.wait(METRIC_SAMPLE_INTERVAL)

def worker(idx):
    session = requests.Session()
    local_counts = {}
    local_latencies = []
    local_errors = []
    while time.time() < end_time:
        try:
            resp = session.post(url, json=payload, timeout=90)
            local_counts[resp.status_code] = local_counts.get(resp.status_code, 0) + 1
            local_latencies.append(resp.elapsed.total_seconds())
        except Exception as exc:
            local_counts['ERR'] = local_counts.get('ERR', 0) + 1
            local_errors.append(str(exc))
        time.sleep(random.uniform(WAIT_MIN, WAIT_MAX))
    return local_counts, local_latencies, local_errors

metrics_thread = threading.Thread(target=metrics_worker, daemon=True)
metrics_thread.start()

with ThreadPoolExecutor(max_workers=USER_COUNT) as pool:
    futures = [pool.submit(worker, i) for i in range(USER_COUNT)]
    for fut in as_completed(futures):
        counts, lats, errs = fut.result()
        for status, cnt in counts.items():
            status_counts[status] = status_counts.get(status, 0) + cnt
        latencies.extend(lats)
        errors.extend(errs)

total_requests = sum(status_counts.values())
metrics_stop.set()
metrics_thread.join(timeout=5)
metric_list = list(metric_samples)
print({'total_requests': total_requests, 'status_counts': status_counts, 'duration_s': round(time.time()-start, 2)})
if latencies:
    latencies.sort()
    def pct(p):
        idx = int(len(latencies)*p/100)
        idx = min(max(idx,0), len(latencies)-1)
        return round(latencies[idx], 3)
    print({'latency_avg': round(sum(latencies)/len(latencies),3), 'p50': pct(50), 'p95': pct(95), 'p99': pct(99)})
if errors:
    from collections import Counter
    print('error_examples', Counter(errors).most_common(5))
if metric_list:
    max_active = max((snap.get('active', 0) or 0) for snap in metric_list)
    max_active_jobs = max((snap.get('active_jobs', 0) or 0) for snap in metric_list)
    avg_wait = sum((snap.get('avg_wait_ms', 0.0) or 0.0) for snap in metric_list) / len(metric_list)
    avg_exec = sum((snap.get('avg_exec_ms', 0.0) or 0.0) for snap in metric_list) / len(metric_list)
    max_wait = max((snap.get('max_wait_ms', 0) or 0) for snap in metric_list)
    loop_avg = sum((snap.get('event_loop_lag_avg_ms', 0.0) or 0.0) for snap in metric_list) / len(metric_list)
    cpu_values = [snap.get('cpu_pct') for snap in metric_list if snap.get('cpu_pct') is not None]
    avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else None
    print({'metrics_samples': len(metric_list), 'max_active_metric': max_active, 'max_active_jobs': max_active_jobs, 'avg_wait_ms_sampled': round(avg_wait, 2), 'max_wait_ms_sampled': max_wait, 'avg_exec_ms_sampled': round(avg_exec, 2), 'avg_loop_lag_ms_sampled': round(loop_avg, 3), 'avg_cpu_pct_sampled': round(avg_cpu, 2) if avg_cpu is not None else None})
