# MLflow + MinIO Backend Unification Progress Report

**Date:** November 3, 2025  
**Status:** Configuration Complete, Deployment Blocked by Init Container Dependencies

---

## Executive Summary

Successfully designed and configured unified backend architecture allowing Kubernetes workloads to connect to Docker's MLflow and MinIO services via `host.docker.internal`. Configuration changes complete in Helm values and templates, but deployment blocked by hardcoded init container dependencies requiring refactoring.

---

## 1. Current State Analysis

### Docker Backend (Canonical Source of Truth)
- **MLflow**: `localhost:5000` → Accessible from K8s as `host.docker.internal:5000`
- **MinIO**: `localhost:9000` → Accessible from K8s as `host.docker.internal:9000`  
- **FastAPI**: `localhost:8000` → Accessible from K8s as `host.docker.internal:8000`
- **Postgres**: Internal to Docker network `flts-main_app-network` (172.18.0.0/16)
- **Kafka**: Docker internal (K8s will use separate instance)

### Kubernetes Before Unification
- Separate MLflow pod with separate Postgres backend
- Separate MinIO pod with 20Gi PVC
- Separate FastAPI pod
- All services ClusterIP with internal DNS (`http://mlflow:5000`, etc.)
- **Result**: Isolated storage, required port-forwarding for cross-environment access

### Connectivity Verification ✅
```bash
# Tested from K8s pod
host.docker.internal (192.168.65.254:5000) open  # MLflow
host.docker.internal (192.168.65.254:9000) open  # MinIO
```
**Confirmed**: K8s pods can reach Docker services on Docker Desktop.

---

## 2. Configuration Changes Applied

### A. Helm Values (`values-complete.yaml`)

#### Added External Backend Section
```yaml
externalBackend:
  enabled: true
  mlflow:
    host: "host.docker.internal"
    port: 5000
    trackingUri: "http://host.docker.internal:5000"
  minio:
    host: "host.docker.internal"
    port: 9000
    endpoint: "http://host.docker.internal:9000"
    accessKey: "minioadmin"
    secretKey: "minioadmin"
  fastapi:
    host: "host.docker.internal"
    port: 8000
    gatewayUrl: "http://host.docker.internal:8000"
```

#### Disabled In-Cluster Services
```yaml
minio:
  enabled: false  # Using external Docker MinIO
  persistence:
    enabled: false
  initJob:
    enabled: false

postgres:
  enabled: false  # Using external Docker Postgres
  persistence:
    enabled: false

mlflow:
  enabled: false  # Using external Docker MLflow

fastapi:
  enabled: false  # Using external Docker FastAPI
```

### B. Helm Templates (`_helpers.tpl`)

#### Added Helper Functions for Dynamic Backend Resolution
```go-template
{{- define "chart.mlflowTrackingUri" -}}
{{- if .Values.externalBackend.enabled }}
{{- .Values.externalBackend.mlflow.trackingUri }}
{{- else }}
{{- printf "http://mlflow:%v" .Values.mlflow.service.port }}
{{- end }}
{{- end }}

{{- define "chart.minioEndpoint" -}}
{{- if .Values.externalBackend.enabled }}
{{- .Values.externalBackend.minio.endpoint }}
{{- else }}
{{- printf "http://minio:%v" .Values.minio.service.apiPort }}
{{- end }}
{{- end }}

# Similar helpers for: minioAccessKey, minioSecretKey, gatewayUrl
```

### C. Updated Templates to Use Helpers

#### Files Modified:
1. `.helm/templates/training-services.yaml`
   - Updated GRU trainer (lines 64-80)
   - Updated LSTM trainer (lines 191-207)
   - Updated Prophet trainer (lines 318-334)
   - Updated Eval service (lines 476-492)

2. `.helm/templates/pipeline.yaml`
   - Updated Preprocess job (lines 115-117)
   - Updated generic Train deployment (lines 217-233)
   - Updated NonML deployment (lines 322-334)
   - Updated Inference deployment (lines 454-466)

#### Example Change:
**Before:**
```yaml
- name: MLFLOW_TRACKING_URI
  value: "http://mlflow:{{ .Values.mlflow.service.port }}"
- name: MLFLOW_S3_ENDPOINT_URL
  value: "http://minio:{{ .Values.minio.service.apiPort }}"
```

**After:**
```yaml
- name: MLFLOW_TRACKING_URI
  value: {{ include "chart.mlflowTrackingUri" . | quote }}
- name: MLFLOW_S3_ENDPOINT_URL
  value: {{ include "chart.minioEndpoint" . | quote }}
```

---

## 3. Deployment Blockers

### Issue: Init Container Circular Dependencies

**Problem**: Helm templates contain hardcoded init containers that wait for in-cluster services:

```yaml
initContainers:
  - name: wait-for-mlflow
    image: busybox:1.35
    command: ['sh', '-c', 'until nc -z mlflow 5000; do sleep 2; done']
  
  - name: wait-for-fastapi
    image: busybox:1.35
    command: ['sh', '-c', 'until nc -z fastapi-app 8000; do sleep 2; done']
```

**Impact**:
- With `mlflow.enabled: false`, no MLflow service/pod created
- Init containers wait forever for non-existent `mlflow:5000` DNS entry
- All dependent pods stuck in `Init:0/2` state
- Helm install times out after 10 minutes

**Observed State**:
```
NAME                           READY   STATUS        
train-gru-fb7df9fb4-tgv2m      0/1     Init:0/2     # Waiting for MLflow
train-lstm-d6d7b4c5b-pqhjp     0/1     Init:0/2     # Waiting for MLflow  
nonml-prophet-96c584db4-89k2v  0/1     Init:0/2     # Waiting for MLflow
eval-d45dd6559-cdl5h           0/1     Init:0/3     # Waiting for MLflow + FastAPI
inference-6b6b6574c9-427pl     0/1     Init:1/2     # Waiting for MLflow
preprocess-043uf-r6fls         0/1     Init:0/2     # Waiting for FastAPI
```

**Files Containing Init Containers**:
1. `.helm/templates/training-services.yaml` (lines 28-48 per service)
2. `.helm/templates/pipeline.yaml` (lines 40-60, 170-190, 415-435)

---

## 4. Solutions (Next Steps)

### Option A: Conditional Init Containers (Recommended)
Wrap init containers with conditional logic:

```yaml
{{- if not .Values.externalBackend.enabled }}
initContainers:
  - name: wait-for-mlflow
    # ... existing config
{{- end }}
```

**Pros**: Clean, preserves functionality for in-cluster mode  
**Cons**: Requires edits to 10+ template sections  
**Effort**: ~30 minutes

### Option B: Remove Init Containers for External Mode
Create external backend templates without init containers:

```yaml
{{- if .Values.externalBackend.enabled }}
# No init containers
{{- else }}
initContainers:
  # ... existing
{{- end }}
```

**Pros**: Clear separation of concerns  
**Cons**: More template duplication  
**Effort**: ~45 minutes

### Option C: Stub Services (Quick Workaround)
Deploy minimal stub services that pods can resolve but don't use:

```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
spec:
  type: ExternalName
  externalName: host.docker.internal
  ports:
    - port: 5000
```

**Pros**: No template changes, init containers pass  
**Cons**: Confusing DNS resolution, not semantically correct  
**Effort**: ~10 minutes

### Option D: Alternative Values File (Created)
Created `.helm/values-external-backend.yaml` with minimal config disabling problematic components.

**Limitation**: Still requires init container fixes to work.

---

## 5. Verification Plan (Post-Fix)

Once init container issues resolved:

### Step 1: Deploy Unified Backend
```bash
helm uninstall flts
helm install flts .helm -f .helm/values-complete.yaml --timeout 10m
```

### Step 2: Verify Pod Startup
```bash
kubectl get pods -w
# Expected: All training/eval/inference pods Running
```

### Step 3: Run 50-Row Pipeline
```bash
# Trigger preprocess via Helm or kubectl
kubectl logs -f job/preprocess-xxxxx

# Verify training completes
kubectl logs -f deployment/train-gru | grep -E "train_complete|run_id"
```

### Step 4: Check Shared MLflow UI
```bash
# Open Docker MLflow (already running)
http://localhost:5000

# Should see new K8s runs appear instantly without port-forwarding
```

### Step 5: Verify MinIO Artifacts
```bash
# Check artifacts uploaded to shared bucket
docker compose exec minio mc ls minio/mlflow/

# Should show artifacts from both Docker and K8s runs
```

### Step 6: Deploy and Scale Inference
```bash
kubectl scale deployment/inference --replicas=5
kubectl get hpa inference  # If HPA configured
kubectl logs -f deployment/inference | grep "model_loaded"
```

### Step 7: Load Test Unified Inference
```bash
# Use Locust or direct requests
for i in {1..100}; do
  curl -X POST http://<inference-lb>/predict \
    -H "Content-Type: application/json" \
    -d @payload-valid.json
done
```

---

## 6. Expected Benefits Post-Unification

### Eliminated Port-Forwarding
**Before**: `kubectl port-forward svc/mlflow 5000:5000` required  
**After**: K8s workloads access `host.docker.internal:5000` directly

### Shared Experiment Tracking
**Before**: Separate MLflow instances, manual export/import  
**After**: Single source of truth at `http://localhost:5000`

### Unified Artifact Storage
**Before**: Docker artifacts in Docker volume, K8s artifacts in PVC  
**After**: All artifacts in Docker MinIO at `localhost:9000/mlflow`

### Simplified Development Workflow
**Before**: Run pipeline in Docker, separately in K8s, compare manually  
**After**: Run anywhere, view results in single MLflow UI

### Consistent Config Hash
Both environments use same backend → same config hash → easier comparison

---

## 7. Configuration Files Summary

### Modified Files
1. `.helm/values-complete.yaml` - Added externalBackend section, disabled services
2. `.helm/templates/_helpers.tpl` - Added 5 helper functions for backend URL resolution
3. `.helm/templates/training-services.yaml` - Updated 4 service env sections
4. `.helm/templates/pipeline.yaml` - Updated 4 deployment env sections

### Created Files
1. `.helm/values-external-backend.yaml` - Minimal values for external mode (blocked by init containers)

### Files Needing Updates (Blockers)
1. `.helm/templates/training-services.yaml` - Init containers (lines 28-48, repeated for each trainer)
2. `.helm/templates/pipeline.yaml` - Init containers (lines 40-60, 88-108, 170-190, 280-300, 415-435)

---

## 8. Recommendations

### Immediate Action
**Complete init container refactoring using Option A (conditional)**:
- Wrap all `wait-for-mlflow` init containers with `{{- if not .Values.externalBackend.enabled }}`
- Wrap all `wait-for-fastapi` init containers with `{{- if not .Values.externalBackend.enabled }}`
- Keep `wait-for-kafka` init containers (Kafka remains in K8s cluster)

### Deploy and Validate
1. Deploy with unified backend
2. Run 50-row sampled pipeline
3. Confirm runs appear in Docker MLflow UI without port-forwarding
4. Scale inference to 5 replicas
5. Run load test and measure performance

### Production Considerations
1. **Security**: Change MinIO credentials from `minioadmin` to secure values
2. **Persistence**: Ensure Docker volumes mounted for MLflow/MinIO data
3. **Networking**: For non-Docker-Desktop clusters, replace `host.docker.internal` with actual service endpoints
4. **Monitoring**: Add Prometheus metrics for unified backend health
5. **Backup**: Regular backups of shared Postgres and MinIO data

---

## 9. Docker vs Kubernetes After Unification

| Aspect | Docker (Before) | Kubernetes (Before) | **Unified (Target)** |
|--------|-----------------|---------------------|----------------------|
| **MLflow** | localhost:5000 | port-forward required | **Both use localhost:5000** |
| **MinIO** | localhost:9000 | port-forward required | **Both use localhost:9000** |
| **Experiments** | Separate DB | Separate DB | **Shared Postgres DB** |
| **Artifacts** | Docker volume | K8s PVC | **Shared MinIO bucket** |
| **Config Hash** | 8999be31... | 6e59091b... | **Same hash** |
| **Port Forwarding** | Not needed | Required | **Eliminated** |
| **UI Access** | Direct | Via kubectl | **Direct for both** |

---

## 10. Next Session Checklist

- [ ] Apply Option A: Add conditionals to init containers in templates
- [ ] Test Helm install with external backend enabled
- [ ] Verify all pods reach Running state
- [ ] Run preprocess job and confirm 50-row sampling
- [ ] Verify training pods complete and log to shared MLflow
- [ ] Check Docker MLflow UI shows K8s runs without port-forward
- [ ] Deploy inference service with 2 replicas
- [ ] Scale to 5 replicas and verify all load same model
- [ ] Run load test (100 requests) and measure p95 latency
- [ ] Generate final unification report with metrics

---

## Conclusion

**Configuration: ✅ Complete**  
**Deployment: ⚠️ Blocked by init containers**  
**Effort to Unblock: ~30 minutes** (Option A refactoring)

All backend URL resolution logic is in place via Helm helpers. Environment variables correctly reference `host.docker.internal` when `externalBackend.enabled: true`. Once init container dependencies are resolved, the unified backend will allow seamless operation across Docker and Kubernetes with zero port-forwarding and shared experiment tracking.

**Ready for**: Init container refactoring → deployment → validation → scaling analysis → final report.
