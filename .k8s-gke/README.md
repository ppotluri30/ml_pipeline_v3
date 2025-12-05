# FLTS ML Pipeline - GKE Deployment

GKE-compatible Kubernetes manifests for the FLTS ML Pipeline.

**No cluster-admin required** - uses standard CPU-based HPA for autoscaling.

## Cluster Information

| Property | Value |
|----------|-------|
| **Cluster Name** | `aiml-dev-xhckg-gke-cluster` |
| **Region** | `europe-west3` |
| **Project ID** | `saas-nokia-an-rnd` |
| **Artifact Registry** | `europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml` |

## Prerequisites

1. GKE cluster running Kubernetes 1.31+
2. Artifact Registry **writer** permissions (already configured)
3. Namespace-level permissions to create Deployments, Services, HPAs, Jobs, ConfigMaps, Secrets, PVCs

**Note:** No KEDA or Prometheus required. Autoscaling uses built-in CPU-based HPA.

## Deploy

```bash
# Authenticate
gcloud auth login
gcloud auth configure-docker europe-west3-docker.pkg.dev

# Get cluster credentials
gcloud container clusters get-credentials aiml-dev-xhckg-gke-cluster --region europe-west3 --project saas-nokia-an-rnd

# Push images to Artifact Registry (you have writer permissions)
REGISTRY="europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml"
for img in inference-http inference-worker train nonml eval preprocess mlflow fastapi-app; do
  docker tag $img:latest $REGISTRY/$img:latest
  docker push $REGISTRY/$img:latest
done

# Deploy the full stack
kubectl apply -k .k8s-gke/

# Verify deployment
kubectl get pods
kubectl get svc
kubectl get hpa
```

## Autoscaling

This deployment uses **CPU-based Horizontal Pod Autoscaler (HPA)** instead of KEDA/Prometheus-based autoscaling:

| Deployment | Min | Max | CPU Target |
|------------|-----|-----|------------|
| inference-http | 2 | 5 | 70% |
| inference-worker | 1 | 3 | 70% |

Check HPA status:
```bash
kubectl get hpa
kubectl describe hpa inference-http-hpa
```

The HPA automatically scales pods up when CPU utilization exceeds 70% and scales down after a 5-minute stabilization window.

## Run Pipeline

```bash
# Start a preprocess job (triggers the full ML pipeline)
kubectl create job preprocess-run-$(date +%Y%m%d%H%M%S) --from=job/preprocess

# Monitor training
kubectl logs -f deployment/train-gru
kubectl logs -f deployment/train-lstm
kubectl logs -f deployment/nonml-prophet

# Check eval promotion
kubectl logs -f deployment/eval
```

## External Access

After deployment, get the external IPs:

```bash
# Inference API
kubectl get svc inference-http -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# MLflow UI
kubectl get svc mlflow -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# MinIO Console
kubectl get svc minio-console -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Locust UI
kubectl get svc locust-master -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

