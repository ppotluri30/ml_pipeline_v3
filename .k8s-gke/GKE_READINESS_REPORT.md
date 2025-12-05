# GKE Readiness Report - FLTS ML Pipeline

**Generated:** 2025-12-03  
**Status:** ✅ READY FOR DEPLOYMENT (No cluster-admin required)

---

## Cluster Configuration

| Property | Value |
|----------|-------|
| **Cluster Name** | `aiml-dev-xhckg-gke-cluster` |
| **Region** | `europe-west3` |
| **Project ID** | `saas-nokia-an-rnd` |
| **Artifact Registry** | `europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml` |

---

## Validation Checklist

### 1. ✅ Image References Updated

All 8 custom images configured for GKE Artifact Registry:

| Image | Registry Path |
|-------|---------------|
| inference-http | `europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml/inference-http:latest` |
| inference-worker | `europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml/inference-worker:latest` |
| train | `europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml/train:latest` |
| nonml | `europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml/nonml:latest` |
| eval | `europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml/eval:latest` |
| preprocess | `europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml/preprocess:latest` |
| mlflow | `europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml/mlflow:latest` |
| fastapi-app | `europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml/fastapi-app:latest` |

### 2. ✅ ImagePullPolicy Configuration

| Service Type | Policy | Status |
|--------------|--------|--------|
| Custom images (8) | `Always` | ✅ Correct |
| Standard images (locust, kafka, postgres, minio) | `IfNotPresent` | ✅ Correct |

### 3. ✅ Service Types Configured

**LoadBalancer (External Access):**
- `inference-http` - Inference API (port 8000)
- `mlflow` - MLflow UI (port 5000)
- `minio-console` - MinIO Console (port 9001)
- `locust-master` - Locust UI (port 8089)

**ClusterIP (Internal):**
- `kafka` - Kafka broker (port 9092)
- `minio` - MinIO API (port 9000)
- `mlflow-postgres` - PostgreSQL (port 5432)
- `fastapi-app` - Gateway (port 8000)
- `eval` - Eval service (port 8050)

### 4. ✅ StorageClass Configuration

All PersistentVolumeClaims use `standard-rwo` (GKE default):
- `minio-data` - 20Gi
- `mlflow-postgres-data` - 5Gi (via StatefulSet volumeClaimTemplate)

### 5. ✅ Internal DNS Names Verified

| Service | DNS Name | Port |
|---------|----------|------|
| Kafka | `kafka:9092` | 9092 |
| MinIO | `minio:9000` | 9000 |
| MLflow | `mlflow:5000` | 5000 |
| FastAPI Gateway | `fastapi-app:8000` | 8000 |

### 6. ✅ CPU-Based HPA Configuration (No KEDA Required)

```yaml
# inference-http-hpa
Target: inference-http
Min Replicas: 2
Max Replicas: 5
CPU Target: 70% averageUtilization
Scale-down stabilization: 300s

# inference-worker-hpa
Target: inference-worker  
Min Replicas: 1
Max Replicas: 3
CPU Target: 70% averageUtilization
Scale-down stabilization: 300s
```

### 7. ✅ No Cluster-Admin Resources

**Removed (not required):**
- ❌ KEDA ScaledObject
- ❌ Prometheus deployment
- ❌ ClusterRole / ClusterRoleBinding
- ❌ Role / RoleBinding

**Uses only namespace-scoped resources:**
- ✅ Deployments
- ✅ StatefulSets
- ✅ Services
- ✅ ConfigMaps
- ✅ Secrets
- ✅ PersistentVolumeClaims
- ✅ Jobs
- ✅ HorizontalPodAutoscalers

### 8. ✅ Kustomize Build Validated

```
Total Resources: 32
Build Status: SUCCESS
```

**Resource Types:**
- ConfigMap (3)
- Deployment (10)
- HorizontalPodAutoscaler (2)
- Job (2)
- PersistentVolumeClaim (1)
- Secret (1)
- Service (7)
- StatefulSet (2)

---

## Deployment Commands

```bash
# 1. Authenticate to GKE
gcloud container clusters get-credentials aiml-dev-xhckg-gke-cluster \
  --region europe-west3 \
  --project saas-nokia-an-rnd

# 2. Configure Docker for Artifact Registry (you have writer permissions)
gcloud auth configure-docker europe-west3-docker.pkg.dev

# 3. Tag and push images
REGISTRY="europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml"
for img in inference-http inference-worker train nonml eval preprocess mlflow fastapi-app; do
  docker tag $img:latest $REGISTRY/$img:latest
  docker push $REGISTRY/$img:latest
done

# 4. Deploy FLTS ML Pipeline (no KEDA or cluster-admin needed)
kubectl apply -k .k8s-gke/

# 5. Verify deployment
kubectl get pods
kubectl get svc
kubectl get hpa

# 6. Get external IPs
kubectl get svc inference-http mlflow minio-console locust-master \
  -o custom-columns='NAME:.metadata.name,EXTERNAL-IP:.status.loadBalancer.ingress[0].ip,PORT:.spec.ports[0].port'
```

---

## Post-Deployment Verification

```bash
# 1. Check all pods are running
kubectl get pods -w

# 2. Verify Kafka is ready
kubectl logs statefulset/kafka | grep -i "started"

# 3. Check MinIO buckets initialized
kubectl logs job/minio-init

# 4. Verify MLflow is accessible
kubectl port-forward svc/mlflow 5000:5000 &
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# 5. Run pipeline
kubectl create job preprocess-run-$(date +%s) --from=job/preprocess

# 6. Monitor training
kubectl logs -f deployment/train-gru
kubectl logs -f deployment/train-lstm
kubectl logs -f deployment/nonml-prophet

# 7. Check eval completion
kubectl logs deployment/eval | grep -i "promotion"

# 8. Test inference
INFERENCE_IP=$(kubectl get svc inference-http -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -X POST http://$INFERENCE_IP:8000/predict -H "Content-Type: application/json" -d '{}'

# 9. Check HPA scaling
kubectl get hpa
kubectl describe hpa inference-http-hpa
```

---

## File Structure

```
.k8s-gke/
├── kustomization.yaml          # Main orchestrator with image transformers
├── README.md                   # Deployment instructions
├── GKE_READINESS_REPORT.md     # This report
├── autoscaling/
│   ├── hpa-inference-http.yaml   # CPU-based HPA (2-5 replicas)
│   └── hpa-inference-worker.yaml # CPU-based HPA (1-3 replicas)
├── backend/
│   ├── fastapi-app.yaml        # MinIO gateway
│   ├── kafka.yaml              # Kafka StatefulSet (KRaft mode)
│   ├── minio.yaml              # MinIO + Console
│   ├── minio-init-job.yaml     # Bucket initialization
│   └── mlflow.yaml             # MLflow + PostgreSQL
├── inference/
│   ├── deployment.yaml         # HTTP inference server
│   └── service.yaml            # LoadBalancer service
├── locust/
│   ├── configmap.yaml          # Locust test script
│   ├── master.yaml             # Locust master
│   └── worker.yaml             # Locust workers
├── pipeline/
│   ├── eval.yaml               # Model evaluation
│   ├── preprocess.yaml         # Data preprocessing Job
│   └── training.yaml           # GRU, LSTM, Prophet trainers
└── worker/
    └── deployment.yaml         # Kafka inference worker
```

---

## Summary

| Requirement | Status |
|-------------|--------|
| Image references point to Artifact Registry | ✅ |
| ImagePullPolicy: Always for custom images | ✅ |
| LoadBalancer for external services | ✅ |
| ClusterIP for internal services | ✅ |
| StorageClass: standard-rwo | ✅ |
| Internal DNS names correct | ✅ |
| CPU-based HPA configured | ✅ |
| No cluster-admin required | ✅ |
| No KEDA/Prometheus required | ✅ |
| Kustomize build validates | ✅ |

**The `.k8s-gke/` folder is ready for deployment to GKE cluster `aiml-dev-xhckg-gke-cluster` with namespace-level permissions only.**

