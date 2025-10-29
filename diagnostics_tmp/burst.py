import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

with open('payload-valid.json','r') as f:
    payload = json.load(f)

url = 'http://localhost:8023/predict'

start = time.time()
results = []

def send(i):
    try:
        r = requests.post(url, json=payload, timeout=30)
        return i, r.status_code, r.elapsed.total_seconds()
    except Exception as exc:
        return i, getattr(getattr(exc, 'response', None), 'status_code', 'ERR'), str(exc)

with ThreadPoolExecutor(max_workers=200) as pool:
    futures = [pool.submit(send, i) for i in range(200)]
    for fut in as_completed(futures):
        results.append(fut.result())

total = time.time() - start
status_counts = {}
for _, status, _ in results:
    status_counts[status] = status_counts.get(status, 0) + 1
print({'total_requests': len(results), 'duration_s': round(total, 2), 'status_counts': status_counts})

slow = sorted(results, key=lambda item: item[2] if isinstance(item[2], (int,float)) else 0, reverse=True)[:5]
print('Top5 slow entries:', slow)
