# YAML File Inventory and Classification

## Classification Legend
- **Category A (Required for Production)**: Keep, may need cleanup
- **Category B (Outdated/Redundant)**: Archive, replace with clean version
- **Category C (Dead/Unsafe)**: Archive immediately, breaks autoscaling or architecture

---

## .k8s/ Directory (Primary Kubernetes Manifests)

| File | Category | Action | Reason |
|------|----------|--------|--------|
| `inference-http-deployment.yaml` | A | ✅ KEEP → `.k8s-clean/inference/deployment.yaml` | Current production deployment |
| `inference-http-keda-prometheus.yaml` | A | ✅ KEEP → `.k8s-clean/autoscaling/scaledobject.yaml` | Working KEDA Prometheus scaler |
| `prometheus-configmap-patch.yaml` | A | ✅ KEEP → `.k8s-clean/autoscaling/prometheus-scrape-config.yaml` | Prometheus scrape config for inference |
| `inference-worker-deployment.yaml` | A | ✅ KEEP → `.k8s-clean/worker/deployment.yaml` | Kafka consumer worker |
| `eval.yaml` | A | ✅ KEEP → `.k8s-clean/pipeline/eval.yaml` | Model evaluation service |
| `eval-rbac.yaml` | A | ✅ KEEP → `.k8s-clean/rbac/eval-rbac.yaml` | RBAC for eval service |
| `locust-master.yaml` | A | ✅ KEEP → `.k8s-clean/locust/master.yaml` | Load testing master |
| `locust-worker.yaml` | A | ✅ KEEP → `.k8s-clean/locust/worker.yaml` | Load testing workers |
| `locust-configmap.yaml` | A | ✅ KEEP → `.k8s-clean/locust/configmap.yaml` | Locust test scripts |
| `train-gru.yaml` | A | ✅ KEEP → `.k8s-clean/pipeline/training.yaml` | GRU trainer |
| `train-lstm.yaml` | A | ✅ KEEP → `.k8s-clean/pipeline/training.yaml` | LSTM trainer |
| `inference-deployment.yaml` | B | 📦 ARCHIVE | Old monolithic inference (has Kafka consumer) |
| `inference-service.yaml` | B | 📦 ARCHIVE | Old service (app=inference, not inference-http) |
| `inference-hpa.yaml` | C | 🗑️ ARCHIVE | CPU-based HPA for old inference deployment |
| `inference-http-hpa.yaml` | C | 🗑️ ARCHIVE | CPU-based HPA, conflicts with KEDA |
| `inference-guardrail-hpa.yaml` | C | 🗑️ ARCHIVE | Secondary HPA, conflicts with KEDA |
| `inference-keda-scaledobject.yaml` | C | 🗑️ ARCHIVE | Old KEDA scaler for `inference` deployment |
| `inference-http-service-prometheus.yaml` | B | 📦 ARCHIVE | Includes ServiceMonitor, may conflict |
| `keda-http-inference.yaml` | C | 🗑️ ARCHIVE | KEDA HTTP Add-on HTTPScaledObject (removed) |
| `keda-http-proxy-service.yaml` | C | 🗑️ ARCHIVE | ExternalName service for HTTP Add-on proxy |
| `locust-driver-job.yaml` | B | 📦 ARCHIVE | Old job-based load test approach |
| `locust-driver-job-simple.yaml` | B | 📦 ARCHIVE | Simplified job approach |
| `fastapi-app.yaml` | B | 📦 ARCHIVE | Not needed in clean set (deployed via Helm) |
| `kafka.yaml` | B | 📦 ARCHIVE | Not needed in clean set (deployed via Helm) |
| `minio.yaml` | B | 📦 ARCHIVE | Not needed in clean set (deployed via Helm) |
| `mlflow.yaml` | B | 📦 ARCHIVE | Not needed in clean set (deployed via Helm) |
| `minio-init-job.yaml` | B | 📦 ARCHIVE | One-time init job |
| `preprocess.yaml` | B | 📦 ARCHIVE | Preprocess deployment (deployed via Helm) |
| `prom-config-current.yaml` | B | 📦 ARCHIVE | Old prometheus config dump |
| `prometheus-patch.yaml` | B | 📦 ARCHIVE | Old prometheus patch |
| `prometheus-server-backup.yaml` | B | 📦 ARCHIVE | Backup file |

---

## .kubernetes/ Directory (Kompose-generated, Legacy)

| File | Category | Action | Reason |
|------|----------|--------|--------|
| `inference-deployment.yaml` | C | 🗑️ ARCHIVE | Kompose-generated, wrong labels, has Kafka consumer |
| `inference-service.yaml` | C | 🗑️ ARCHIVE | Wrong selector (io.kompose.service) |
| `inference-deployment-slo.yaml` | C | 🗑️ ARCHIVE | SLO variant, not used |
| `inference-debug-patch.yaml` | B | 📦 ARCHIVE | Debug patch |
| `inference-reduced-resources.yaml` | B | 📦 ARCHIVE | Resource variant |
| `inference-probe-fix.yaml` | B | 📦 ARCHIVE | Probe fix patch |
| `inference-keda-scaler.yaml` | C | 🗑️ ARCHIVE | Scaler for wrong deployment |
| `preprocess-deployment.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `preprocess-service.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `train-deployment.yaml` | B | 📦 ARCHIVE | Generic train, replaced by train-gru/lstm |
| `train-gru-deployment.yaml` | B | 📦 ARCHIVE | Duplicate of .k8s/train-gru.yaml |
| `train-service.yaml` | B | 📦 ARCHIVE | Not needed (no external access) |
| `nonml-prophet-deployment.yaml` | A | ✅ Merged into `.k8s-clean/pipeline/training.yaml` | Prophet trainer |
| `eval-deployment.yaml` | B | 📦 ARCHIVE | Kompose version, use .k8s/eval.yaml |
| `eval-service.yaml` | B | 📦 ARCHIVE | Merged into eval.yaml |
| `eda-deployment.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `eda-service.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `kafka-deployment.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `kafka-service.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `minio-deployment.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `minio-service.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `minio-data-persistentvolumeclaim.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `mlflow-deployment.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `mlflow-service.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `postgres-deployment.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `postgres-service.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `fastapi-app-deployment.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `fastapi-app-service.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `fastapi-app-claim0-persistentvolumeclaim.yaml` | B | 📦 ARCHIVE | Deployed via Helm |
| `env-minio-configmap.yaml` | B | 📦 ARCHIVE | Legacy config |

---

## Root Directory (Scattered YAMLs)

| File | Category | Action | Reason |
|------|----------|--------|--------|
| `current-scaledobject.yaml` | C | 🗑️ ARCHIVE | Dump of old scaledobject, targets wrong deployment |
| `keda-inference-http-redirect.yaml` | C | 🗑️ ARCHIVE | ExternalName service for HTTP Add-on |
| `keda-backend-service.yaml` | C | 🗑️ ARCHIVE | Backend service for HTTP Add-on |
| `keda-operator-backup.yaml` | B | 📦 ARCHIVE | Backup file |
| `keda-metrics-apiserver-backup.yaml` | B | 📦 ARCHIVE | Backup file |
| `prometheus-server-backup.yaml` | B | 📦 ARCHIVE | Backup file |
| `prometheus-config-backup.yaml` | B | 📦 ARCHIVE | Backup file |
| `prometheus-config-full.yaml` | B | 📦 ARCHIVE | Full config dump |
| `prometheus-inference-fast-scrape.yaml` | B | 📦 ARCHIVE | Superseded by prometheus-scrape-config.yaml |
| `prom-config-edit.yaml` | B | 📦 ARCHIVE | Edit file |
| `preprocess-job-template.yaml` | B | 📦 ARCHIVE | Job template |
| `preprocess-job-manual.yaml` | B | 📦 ARCHIVE | Manual job |
| `values-lowpower.yaml` | B | 📦 ARCHIVE | Helm values variant |
| `docker-compose.yaml` | A | ✅ KEEP | Docker Compose for local dev |
| `docker-compose.staging.yaml` | A | ✅ KEEP | Staging config |
| `docker-compose.override.yaml` | A | ✅ KEEP | Override config |

---

## .helm/ Directory (Keep as-is)

All Helm chart files should be kept - they are the source of truth for non-inference infrastructure:
- `Chart.yaml`, `values*.yaml`
- `templates/*.yaml`

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| **A - Required** | 14 | Consolidated into `.k8s-clean/` |
| **B - Outdated** | 35 | Moved to `.k8s-archive/` |
| **C - Dead/Unsafe** | 12 | Moved to `.k8s-archive/` |

---

## Clean Directory Structure

```
.k8s-clean/
├── kustomization.yaml           # Deploy all with: kubectl apply -k .k8s-clean/
├── namespace.yaml               # Namespace definition
├── inference/
│   ├── deployment.yaml          # inference-http Deployment
│   └── service.yaml             # inference-http Service
├── worker/
│   └── deployment.yaml          # inference-worker Deployment
├── autoscaling/
│   ├── scaledobject.yaml        # KEDA ScaledObject (Prometheus triggers)
│   └── prometheus-scrape-config.yaml  # Prometheus ConfigMap patch
├── locust/
│   ├── configmap.yaml           # Locust test scripts
│   ├── master.yaml              # Locust master Deployment + Service
│   └── worker.yaml              # Locust worker Deployment
├── rbac/
│   └── eval-rbac.yaml           # ServiceAccount + Role + RoleBinding
└── pipeline/
    ├── eval.yaml                # Eval Deployment + Service
    └── training.yaml            # GRU, LSTM, Prophet Deployments
```
