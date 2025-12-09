# Kubernetes Locust Architecture Diagrams

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           Kubernetes Cluster (default namespace)                     │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  Locust Master (Deployment: 1 replica)                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Pod: locust-master-xxx                                              │  │   │
│  │  │  Image: locustio/locust:latest                                       │  │   │
│  │  │  Env: LOCUST_MODE=master, TARGET_HOST=http://inference:8000          │  │   │
│  │  │                                                                       │  │   │
│  │  │  Ports:                                                               │  │   │
│  │  │    8089  → HTTP API + Web UI                                         │  │   │
│  │  │    5557  → Master communication (P1)                                 │  │   │
│  │  │    5558  → Master communication (P2)                                 │  │   │
│  │  │                                                                       │  │   │
│  │  │  Volume: /home/locust (ConfigMap: locust-scripts)                   │  │   │
│  │  └──────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                               │                                                     │
│                               │ Service: locust-master                              │
│                               │ Type: NodePort (30089)                              │
│                               │                                                     │
│    ┌──────────────────────────┼──────────────────────────┐                         │
│    │                          │                          │                         │
│    │                          ▼                          │                         │
│    │      ┌──────────────────────────────────┐          │                         │
│    │      │  Service: locust-master          │          │                         │
│    │      │  ClusterIP: 10.x.x.x             │          │                         │
│    │      │  Ports: 8089, 5557, 5558         │          │                         │
│    │      └──────────────────────────────────┘          │                         │
│    │                    │                                │                         │
│    │                    │ TCP                            │                         │
│    │                    ▼                                │                         │
│    │      ┌────────────────────────────────────┐        │                         │
│    │      │  Locust Workers (Deployment)       │        │                         │
│    │      │  Replicas: 4 (scalable to 8+)     │        │                         │
│    │      │                                    │        │                         │
│    │      │  ┌────────────────────────────┐   │        │                         │
│    │      │  │ Pod: locust-worker-xxx-1   │   │        │                         │
│    │      │  │ LOCUST_MODE=worker         │   │        │                         │
│    │      │  │ MASTER=locust-master:5557  │   │        │                         │
│    │      │  └────────────────────────────┘   │        │                         │
│    │      │  ┌────────────────────────────┐   │        │                         │
│    │      │  │ Pod: locust-worker-xxx-2   │   │        │                         │
│    │      │  └────────────────────────────┘   │        │                         │
│    │      │  ┌────────────────────────────┐   │        │                         │
│    │      │  │ Pod: locust-worker-xxx-3   │   │        │                         │
│    │      │  └────────────────────────────┘   │        │                         │
│    │      │  ┌────────────────────────────┐   │        │                         │
│    │      │  │ Pod: locust-worker-xxx-4   │   │        │                         │
│    │      │  └────────────────────────────┘   │        │                         │
│    │      └────────────────────────────────────┘        │                         │
│    │                                                     │                         │
│    │         Workers generate load → Target             │                         │
│    │                                  ▼                  │                         │
│    │                    ┌──────────────────────────────┐│                         │
│    │                    │  Service: inference           ││                         │
│    │                    │  ClusterIP: 10.107.123.158   ││                         │
│    │                    │  Port: 8000                   ││                         │
│    │                    └──────────────────────────────┘│                         │
│    │                                  │                  │                         │
│    │                                  ▼                  │                         │
│    │               ┌───────────────────────────────────┐│                         │
│    │               │  Inference Deployment              ││                         │
│    │               │  Replicas: 1-8 (scaled by driver) ││                         │
│    │               │                                    ││                         │
│    │               │  ┌──────────────────────────┐     ││                         │
│    │               │  │ Pod: inference-xxx-1     │     ││                         │
│    │               │  │ Image: inference:latest  │     ││                         │
│    │               │  │ Resources: 1CPU, 2Gi RAM │     ││                         │
│    │               │  └──────────────────────────┘     ││                         │
│    │               │  ┌──────────────────────────┐     ││                         │
│    │               │  │ Pod: inference-xxx-2     │     ││                         │
│    │               │  └──────────────────────────┘     ││                         │
│    │               └───────────────────────────────────┘│                         │
│    │                                                     │                         │
│    └─────────────────────────────────────────────────────┘                         │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  Driver Job (Orchestration)                                                  │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐   │  │
│  │  │  Pod: locust-load-test-matrix-xxx                                    │   │  │
│  │  │  Image: locust-driver:latest                                         │   │  │
│  │  │  ServiceAccount: locust-driver (RBAC: scale deployments)             │   │  │
│  │  │                                                                       │   │  │
│  │  │  1. Scale inference deployment (kubectl scale)                       │   │  │
│  │  │     Replicas: [1, 2, 4, 8]                                           │   │  │
│  │  │                                                                       │   │  │
│  │  │  2. Scale worker deployment (kubectl scale)                          │   │  │
│  │  │     Workers: [4, 8]                                                  │   │  │
│  │  │                                                                       │   │  │
│  │  │  3. Start test via Locust API (POST /swarm)                          │   │  │
│  │  │     Users: [200, 400, 800]                                           │   │  │
│  │  │     Duration: 120s per test                                          │   │  │
│  │  │                                                                       │   │  │
│  │  │  4. Poll stats (GET /stats/requests)                                 │   │  │
│  │  │     Live RPS, latency, failures                                      │   │  │
│  │  │                                                                       │   │  │
│  │  │  5. Stop test (GET /stop)                                            │   │  │
│  │  │                                                                       │   │  │
│  │  │  6. Export results (/results/auto_summary.csv, .md)                  │   │  │
│  │  │                                                                       │   │  │
│  │  │  Total: 24 scenarios (~54 minutes)                                   │   │  │
│  │  └──────────────────────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  ConfigMap: locust-scripts                                                   │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐   │  │
│  │  │  locustfile.py (simplified, 60 lines)                                │   │  │
│  │  │    class InferenceUser(HttpUser):                                    │   │  │
│  │  │      @task(8) predict():        POST /predict (80%)                  │   │  │
│  │  │      @task(1) health_check():   GET /healthz (10%)                   │   │  │
│  │  │      @task(1) metrics_check():  GET /metrics (10%)                   │   │  │
│  │  │                                                                       │   │  │
│  │  │    Payload: 30 timestamps × 11 features                              │   │  │
│  │  │      [down, up, rnti_count, mcs_down, mcs_up, ...]                   │   │  │
│  │  └──────────────────────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

External Access:
  http://localhost:30089 → Locust Master Web UI + API (NodePort)
```

## Test Flow Sequence

```
┌──────────┐                                                                           
│ Driver   │                                                                           
│   Job    │                                                                           
└────┬─────┘                                                                           
     │                                                                                 
     │ 1. Scale inference (kubectl scale deployment inference --replicas=N)           
     ├────────────────────────────────────────────────────────────────────────────────▶
     │                                                                                 
     │ 2. Wait for readiness (kubectl wait --for=condition=ready)                     
     ├────────────────────────────────────────────────────────────────────────────────▶
     │                                                                                 
     │ 3. Scale workers (kubectl scale deployment locust-worker --replicas=M)         
     ├────────────────────────────────────────────────────────────────────────────────▶
     │                                                                                 
     │ 4. Wait for workers ready                                                      
     ├────────────────────────────────────────────────────────────────────────────────▶
     │                                                                                 
     │ 5. Reset stats (GET /stats/reset)                                              
     ├─────────────────────────────▶ ┌──────────┐                                     
     │                               │  Locust  │                                     
     │ 6. Start test (POST /swarm)   │  Master  │                                     
     ├─────────────────────────────▶ │  (API)   │                                     
     │   {user_count: X,             └────┬─────┘                                     
     │    spawn_rate: Y,                  │                                           
     │    host: inference:8000}           │ 7. Distribute load to workers             
     │                                    ├────────────────────────────────────────────▶
     │                                    │                                            
     │                                    ▼                                            
     │                              ┌──────────┐                                       
     │                              │  Locust  │                                       
     │                              │ Workers  │                                       
     │                              │  (4-8)   │                                       
     │                              └────┬─────┘                                       
     │                                   │                                             
     │                                   │ 8. Generate requests                        
     │                                   ├────────────────────────────────────────────▶
     │                                   │                                             
     │                                   │    POST /predict (80%)                      
     │                                   │    GET /healthz (10%)                       
     │                                   │    GET /metrics (10%)                       
     │                                   ├───────────────────────────────┐             
     │                                   │                               │             
     │                                   ▼                               ▼             
     │                             ┌──────────┐                   ┌──────────┐         
     │                             │Inference │                   │Inference │         
     │                             │  Pod 1   │                   │  Pod 2   │         
     │                             └────┬─────┘                   └────┬─────┘         
     │                                  │                              │               
     │                                  │ 9. Process predictions       │               
     │                                  │    (LSTM model inference)    │               
     │                                  │                              │               
     │                                  │ ◀────────────────────────────┘               
     │                                  │                                              
     │                                  │ 10. Collect metrics at master                
     │                             ┌────┴─────┐                                        
     │                             │  Locust  │                                        
     │                             │  Master  │                                        
     │                             └────┬─────┘                                        
     │                                  │                                              
     │ 11. Poll stats every 2s          │                                              
     │    (GET /stats/requests)         │                                              
     ├──────────────────────────────────┤                                              
     │ ◀────────────────────────────────┤                                              
     │   {current_rps: XX,              │                                              
     │    median_response_time: YY,     │                                              
     │    ninetieth_response_time: ZZ}  │                                              
     │                                  │                                              
     │ 12. Display live progress        │                                              
     │     [60s/120s] RPS: 18.5 |       │                                              
     │     Median: 620ms | P90: 1800ms  │                                              
     │                                  │                                              
     │ ... (repeat polling for 120s)    │                                              
     │                                  │                                              
     │ 13. Stop test (GET /stop)        │                                              
     ├─────────────────────────────────▶│                                              
     │                                  │                                              
     │ 14. Get final stats              │                                              
     ├──────────────────────────────────┤                                              
     │ ◀────────────────────────────────┤                                              
     │                                  │                                              
     │ 15. Record results               │                                              
     │     (CSV row + MD entry)         │                                              
     │                                  │                                              
     │ 16. Cooldown (15s)               │                                              
     ├─────────────────────────────────────────────────────────────────────────────────
     │                                                                                 
     │ 17. Next test scenario (repeat from step 1)                                    
     │     Total: 24 scenarios                                                         
     │                                                                                 
     │ 18. Export results                                                              
     │     /results/auto_summary.csv                                                   
     │     /results/auto_summary.md                                                    
     │                                                                                 
     ▼                                                                                 
   [DONE]                                                                              
```

## Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  Test Payload (Locust Workers → Inference Service)                            │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  POST http://inference:8000/predict                                           │
│  Content-Type: application/json                                               │
│                                                                                │
│  {                                                                             │
│    "data": [                                                                   │
│      {                                                                         │
│        "timestamp": "2024-01-15T10:00:00",                                     │
│        "features": {                                                           │
│          "down": 1234.56,                                                      │
│          "up": 789.01,                                                         │
│          "rnti_count": 15,                                                     │
│          "mcs_down": 18,                                                       │
│          "mcs_up": 12,                                                         │
│          "bler_down": 0.02,                                                    │
│          "bler_up": 0.01,                                                      │
│          "prb_down": 45,                                                       │
│          "prb_up": 30,                                                         │
│          "sinr_down": 22.5,                                                    │
│          "sinr_up": 20.1                                                       │
│        }                                                                       │
│      },                                                                        │
│      ... (29 more timestamps)                                                  │
│    ]                                                                           │
│  }                                                                             │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
              │
              │ Inference processes with LSTM model
              ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  Response (Inference Service → Locust Workers)                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  HTTP 200 OK                                                                   │
│  Content-Type: application/json                                               │
│                                                                                │
│  {                                                                             │
│    "predictions": [0.123, 0.456, 0.789, ...],                                 │
│    "model": "LSTM",                                                            │
│    "run_id": "4a4e0e5182934d0780520ca6f610b9d2",                               │
│    "timestamp": "2024-01-15T10:00:01"                                          │
│  }                                                                             │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
              │
              │ Workers report metrics to master
              ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  Aggregated Stats (Locust Master → Driver)                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  GET http://locust-master:8089/stats/requests                                 │
│                                                                                │
│  {                                                                             │
│    "stats": [                                                                  │
│      {                                                                         │
│        "name": "Aggregated",                                                   │
│        "num_requests": 2250,                                                   │
│        "num_failures": 0,                                                      │
│        "median_response_time": 620,                                            │
│        "ninetieth_response_time": 1800,  // Often p90, not p95                │
│        "current_rps": 18.75,                                                   │
│        "current_fail_per_sec": 0.0                                             │
│      },                                                                        │
│      {                                                                         │
│        "name": "POST /predict",                                                │
│        "num_requests": 1800,                                                   │
│        "median_response_time": 640,                                            │
│        ...                                                                     │
│      },                                                                        │
│      ...                                                                       │
│    ],                                                                          │
│    "user_count": 200,                                                          │
│    "state": "running"                                                          │
│  }                                                                             │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
              │
              │ Driver processes and exports
              ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  Results Output (Driver → /results/)                                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  auto_summary.csv:                                                            │
│  Replicas,Workers,Users,RPS,Median_ms,P95_ms,P99_ms,Failures_Pct,...          │
│  2,4,200,18.75,620,1800,2400,0.00,2250,120,...                                │
│                                                                                │
│  auto_summary.md:                                                             │
│  # Kubernetes Locust Load Test Results                                        │
│  ...                                                                           │
│  - **Best Throughput:** 21.23 RPS with 4 replicas, 8 workers, 800 users      │
│  ...                                                                           │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Resource Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ConfigMap: locust-scripts                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  locustfile.py (test script)                                       │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────┬───────────────────────────────────┘
                     │                    │
                     │ mounted as volume  │ mounted as volume
                     ▼                    ▼
      ┌──────────────────────┐   ┌──────────────────────┐
      │  Deployment:         │   │  Deployment:         │
      │  locust-master       │   │  locust-worker       │
      │  (1 replica)         │   │  (4-8 replicas)      │
      └──────────┬───────────┘   └──────────┬───────────┘
                 │                          │
                 │ Service: locust-master   │
                 │ (ClusterIP + NodePort)   │
                 └───────────┬──────────────┘
                             │
                             │ master-worker communication
                             │ (ports 5557, 5558)
                             │
                             ▼
                   ┌──────────────────────┐
                   │  Workers connect to  │
                   │  locust-master:5557  │
                   └──────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│  ServiceAccount: locust-driver                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  RBAC Permissions                                                  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────────────────────────┘
                     │
                     │ attached via roleBinding
                     ▼
      ┌──────────────────────────────────────────┐
      │  Role: locust-driver-role                │
      │  Rules:                                  │
      │    - apiGroups: ["apps"]                 │
      │      resources: ["deployments",          │
      │                  "deployments/scale"]    │
      │      verbs: ["get", "list",              │
      │               "patch", "update"]         │
      └──────────────────┬───────────────────────┘
                         │
                         │ used by
                         ▼
              ┌──────────────────────┐
              │  Job:                │
              │  locust-load-test-   │
              │  matrix              │
              │                      │
              │  Image:              │
              │  locust-driver:latest│
              └──────────────────────┘
                         │
                         │ kubectl scale deployment ...
                         │ curl locust-master:8089/...
                         ▼
              ┌──────────────────────┐
              │  Scales and controls │
              │  all deployments     │
              └──────────────────────┘
```

## Deployment Order

```
Step 1: RBAC Setup
   └─▶ kubectl apply -f locust-driver-job.yaml (ServiceAccount, Role, RoleBinding)

Step 2: Test Script
   └─▶ kubectl apply -f locust-configmap.yaml (ConfigMap: locust-scripts)

Step 3: Locust Infrastructure
   ├─▶ kubectl apply -f locust-master.yaml (Deployment + Service)
   └─▶ kubectl apply -f locust-worker.yaml (Deployment, 4 replicas)

Step 4: Build Driver Image (optional, recommended)
   └─▶ docker build -t locust-driver:latest -f Dockerfile.driver .

Step 5: Run Load Test
   └─▶ kubectl apply -f locust-driver-job-simple.yaml (Job)

Step 6: Monitor and Collect
   ├─▶ kubectl logs -f job/locust-load-test-matrix
   └─▶ kubectl cp <pod>:/results/auto_summary.csv ./
```

## Network Flow

```
External Client                 Kubernetes Cluster
     │                                │
     │ http://localhost:30089         │
     └───────────────────────────────▶│ NodePort Service
                                      │   locust-master:30089
                                      │         │
                                      │         ▼
                                      │   ClusterIP Service
                                      │   locust-master:8089
                                      │         │
                                      │         ▼
                                      │   Pod: locust-master
                                      │   (8089, 5557, 5558)
                                      │         │
                                      │         │ Workers connect via:
                                      │         │ locust-master:5557
                                      │         ▼
                                      │   Pods: locust-worker-xxx
                                      │   (connects to master)
                                      │         │
                                      │         │ Workers send requests to:
                                      │         │ inference:8000
                                      │         ▼
                                      │   ClusterIP Service
                                      │   inference:8000
                                      │         │
                                      │         ▼
                                      │   Pods: inference-xxx
                                      │   (1-8 replicas)
```

---

**Legend:**
- `→` : Data/request flow
- `▶` : Process/action direction
- `│` : Connection/dependency
- `┌─┐` : Component boundary
- `[...]` : External entity
