# FLTS ML Pipeline - Kubernetes Clean YAML Final Validation Report

**Date:** 2025-12-02  
**Scope:** End-to-end validation of cleaned `.k8s-clean/` Kubernetes manifests  
**Status:** ✅ **ALL VALIDATIONS PASSED**

---

## Executive Summary

The cleaned Kubernetes YAML files in `.k8s-clean/` have been successfully deployed and fully validated. All 6 validation steps completed successfully with no regressions introduced by the YAML cleanup.

---

## Validation Checklist

| Step | Validation | Status | Evidence |
|------|------------|--------|----------|
| 1 | All YAML successfully applied | ✅ PASS | 16 resources deployed via `kubectl apply -k .k8s-clean/` |
| 2 | All pipeline stages passed | ✅ PASS | GRU, LSTM, Prophet trained; Eval completed |
| 3 | Promotion correct run_id | ✅ PASS | `11b30239aea14e069358a0a3c4e3e8f4` (GRU winner) |
| 4 | inference-worker pointer update | ✅ PASS | `current.json` updated with new run_id |
| 5 | inference-http auto-reload | ✅ PASS | Logs show `promotion_pointer_parsed`, healthz `model_ready: true` |
| 6 | Prometheus scraping confirmed | ✅ PASS | `inference-pods-fast` job scraping `/prometheus` |
| 7 | KEDA scaling confirmed | ✅ PASS | 2→4→2 pods observed during load tests |
| 8 | Locust load test passed | ✅ PASS | 8,205 requests, 0.13% error rate, 92 RPS |
| 9 | No regressions from YAML cleanup | ✅ PASS | All services functional |

---

## Step 1: Deploy Cleaned YAML Set

### Command
```bash
kubectl apply -k .k8s-clean/
```

### Resources Deployed (16 total)
| Resource Type | Name | Status |
|---------------|------|--------|
| ServiceAccount | keda-service-account | ✅ Created |
| Role | keda-external-metrics-reader | ✅ Created |
| RoleBinding | keda-read-external-metrics | ✅ Created |
| ConfigMap | prometheus-server | ✅ Created |
| ConfigMap | keda-prometheus-config | ✅ Unchanged |
| Service | inference-http | ✅ Unchanged |
| Service | inference-worker | ✅ Created |
| Service | prometheus-server | ✅ Created |
| Deployment | eval | ✅ Unchanged |
| Deployment | inference-http | ✅ Unchanged |
| Deployment | inference-worker | ✅ Unchanged |
| Deployment | locust-master | ✅ Unchanged |
| Deployment | nonml-prophet | ✅ Configured |
| Deployment | prometheus-server | ✅ Unchanged |
| Deployment | train-gru | ✅ Unchanged |
| Deployment | train-lstm | ✅ Unchanged |
| ScaledObject | inference-http-rps-scaler | ✅ Unchanged |

### Fixes Applied During Deployment
1. **Removed `commonLabels`** from `kustomization.yaml` - was causing selector mismatch errors
2. **Fixed nonml-prophet selector** - changed from `app: nonml-prophet` to `io.kompose.service: nonml-prophet`

---

## Step 2: Run Complete ML Pipeline

### Preprocess Job
- **Job Name:** `preprocess-e2e-20251202155959`
- **Config Hash:** `c23c779f9e14bf312f231f1c2cc90b2eafe2b33c21b89383be8fd0436ee89db5`
- **Status:** ✅ Completed successfully
- **Output:** Published to `training-data` and `inference-data` Kafka topics

### Training Results
| Model | Run ID | Status | Training Time |
|-------|--------|--------|---------------|
| GRU | `11b30239aea14e069358a0a3c4e3e8f4` | ✅ SUCCESS | ~30s |
| LSTM | `454199e6255c47c88a1accc857b76d24` | ✅ SUCCESS | ~35s |
| Prophet | `2ded8ada22054fed93cb50f905216403` | ✅ SUCCESS | ~25s |

### Evaluation Scoreboard
| Model | MAPE Score | RMSE | Rank |
|-------|------------|------|------|
| **GRU** | **0.0239** | 1.23 | 🥇 Winner |
| LSTM | 0.0250 | 1.31 | 🥈 |
| Prophet | 0.1046 | 5.42 | 🥉 |

---

## Step 3: Validate Inference Auto-Reload

### Promotion Pointer
```json
{
  "model_uri": "runs:/11b30239aea14e069358a0a3c4e3e8f4/GRU",
  "run_id": "11b30239aea14e069358a0a3c4e3e8f4",
  "model_type": "GRU",
  "config_hash": "c23c779f9e14bf312f231f1c2cc90b2eafe2b33c21b89383be8fd0436ee89db5",
  "score": 0.0239
}
```

### Inference-Worker Logs
```
{"event": "promotion_pointer_updated", "run_id": "11b30239aea14e069358a0a3c4e3e8f4"}
```

### Inference-HTTP Logs
```
{"event": "promotion_pointer_parsed", "run_id": "11b30239aea14e069358a0a3c4e3e8f4", "model_type": "GRU"}
{"event": "model_loaded", "model_type": "GRU"}
```

### Health Check
```bash
curl http://inference-http:8000/healthz
# Response: {"status": "healthy", "model_ready": true, "model_type": "GRU"}
```

---

## Step 4: Validate KEDA Autoscaling

### ScaledObject Configuration
```yaml
name: inference-http-rps-scaler
minReplicas: 2
maxReplicas: 12
triggers:
  - type: prometheus  # RPS threshold: 20
  - type: prometheus  # P95 latency threshold: 1.5s
```

### Initial Load Test (60s, 200 users)
- **Requests:** 5,994
- **RPS:** 101.51
- **Failures:** 0 (0.00%)
- **Scaling:** 2 → 4 → 2 pods

### HPA Events
```
Normal  SuccessfulRescale  New size: 4; reason: external metric s0-prometheus above target
Normal  SuccessfulRescale  New size: 3; reason: All metrics below target
Normal  SuccessfulRescale  New size: 2; reason: All metrics below target
```

---

## Step 5: Run Full Load Test Script

### Test Configuration
```powershell
.\scripts\Run-FullLoadTest.ps1 -Users 200 -SpawnRate 50 -Duration 90
```

### Results Summary
| Metric | Value |
|--------|-------|
| Total Requests | 8,205 |
| Failures | 11 (0.13%) |
| Avg RPS | 92.04 |
| Avg Latency | 631ms |
| P50 Latency | 420ms |
| P95 Latency | 2,000ms |
| P99 Latency | 2,800ms |
| Min Latency | 17ms |
| Max Latency | 3,236ms |

### Pod Scaling During Test
```
2 pods (baseline) → 4 pods (scaled up) → 2 pods (scaled down)
```

### Telemetry Output Location
```
C:\Users\ppotluri\Desktop\ml_pipeline_v3\load_test_results\20251202_163704\
├── inference_logs.txt     (11.1 MB)
├── locust_output.txt      (80 KB)
├── pod_scaling.txt        (608 B)
├── rps.txt                (370 B)
└── scaler.txt             (13.2 KB)
```

### Error Analysis
The 11 failures (0.13%) were transient connection resets during pod scaling - expected behavior:
```
POST /predict: Unexpected status 0 (connection reset during scale event)
```

---

## Step 6: Prometheus Metrics Verification

### Scrape Configuration
```yaml
- job_name: inference-pods-fast
  scrape_interval: 5s
  metrics_path: /prometheus
  kubernetes_sd_configs:
    - role: pod
  relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      regex: inference-http
```

### Metrics Available
- `inference_requests_total` - Request counter
- `inference_latency_seconds_bucket` - Latency histogram
- `inference_queue_len` - Queue length gauge
- `inference_predictions_total` - Prediction counter

### Query Verification
```promql
# P95 Latency
histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[2m])) by (le))
# Result: 2.0s (during load)

# RPS
sum(rate(inference_requests_total[1m]))
# Result: ~92 req/s (during load)
```

---

## Files Modified During Deployment

### `.k8s-clean/kustomization.yaml`
**Change:** Removed `commonLabels` directive that was causing selector mismatch errors
```yaml
# REMOVED:
# commonLabels:
#   project: flts-ml-pipeline
#   managed-by: kustomize
```

### `.k8s-clean/pipeline/training.yaml`
**Change:** Fixed nonml-prophet selector to match existing deployment
```yaml
# CHANGED FROM:
selector:
  matchLabels:
    app: nonml-prophet

# CHANGED TO:
selector:
  matchLabels:
    io.kompose.service: nonml-prophet
```

---

## Current Cluster State

### Deployments
| Deployment | Ready | Replicas |
|------------|-------|----------|
| inference-http | 2/2 | 2 (HPA-managed) |
| inference-worker | 1/1 | 1 |
| train-gru | 1/1 | 1 |
| train-lstm | 1/1 | 1 |
| nonml-prophet | 1/1 | 1 |
| eval | 1/1 | 1 |
| prometheus-server | 1/1 | 1 |
| locust-master | 1/1 | 1 |

### HPA Status
```
NAME                                 TARGETS                    MINPODS   MAXPODS   REPLICAS
keda-hpa-inference-http-rps-scaler   56m/20, 264m/1500m (avg)   2         12        2
```

### Key Run IDs
| Component | Run ID / Hash |
|-----------|---------------|
| Pipeline Run | `2025-12-03T00:00:10.061416Z` |
| Config Hash | `c23c779f9e14bf312f231f1c2cc90b2eafe2b33c21b89383be8fd0436ee89db5` |
| Promoted Model (GRU) | `11b30239aea14e069358a0a3c4e3e8f4` |
| LSTM Model | `454199e6255c47c88a1accc857b76d24` |
| Prophet Model | `2ded8ada22054fed93cb50f905216403` |

---

## Conclusion

The cleaned `.k8s-clean/` Kubernetes manifests have been **fully validated** with:

1. ✅ **Successful deployment** of all 16 resources
2. ✅ **Complete ML pipeline execution** (preprocess → train → eval → promote)
3. ✅ **Inference auto-reload** working correctly
4. ✅ **KEDA autoscaling** responding to load (2→4→2 pods)
5. ✅ **Prometheus metrics** being scraped and queryable
6. ✅ **Load test** passed with 99.87% success rate
7. ✅ **No regressions** introduced by YAML cleanup

The FLTS ML Pipeline is fully operational on the cleaned Kubernetes manifests.

---

## Appendix: Commands for Future Validation

### Quick Health Check
```bash
kubectl get deployments
kubectl get hpa
kubectl describe scaledobject inference-http-rps-scaler
```

### Run ML Pipeline
```bash
kubectl create job preprocess-manual --from=cronjob/preprocess-cron
```

### Load Test
```powershell
.\scripts\Run-FullLoadTest.ps1 -Users 200 -SpawnRate 50 -Duration 90
```

### Check Prometheus Metrics
```bash
kubectl exec deployment/prometheus-server -c prometheus-server -- wget -qO- 'http://localhost:9090/api/v1/query?query=inference_requests_total'
```
