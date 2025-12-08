# Running the GKE ML Pipeline from Your Laptop

## Overview

This pipeline runs **directly on GKE** (Google Kubernetes Engine). No local Kubernetes environment (Docker Desktop, Minikube, Kind) is required. Your laptop is used only to:

1. Authenticate to GCP
2. Build and push Docker images to Artifact Registry
3. Deploy manifests to the remote GKE cluster
4. Monitor and debug using `kubectl`

---

## Prerequisites

- **gcloud CLI** installed ([Install Guide](https://cloud.google.com/sdk/docs/install))
- **Docker** installed and running
- **kubectl** installed (`gcloud components install kubectl`)

---

## Step-by-Step Deployment

### 1. Authenticate to GCP

```powershell
# Login to your Google account
gcloud auth login

# Set the project
gcloud config set project saas-nokia-an-rnd

# Configure Docker to use gcloud for Artifact Registry
gcloud auth configure-docker europe-west3-docker.pkg.dev
```

### 2. Connect kubectl to the GKE Cluster

```powershell
gcloud container clusters get-credentials aiml-dev-xhckg-gke-cluster --region europe-west3 --project saas-nokia-an-rnd
```

Verify connection:

```powershell
kubectl get nodes
```

### 3. Build Docker Images

From the repository root:

```powershell
# Inference HTTP
docker build -t inference-http:latest ./inference_container

# Inference Worker (same image, different entry point configured in deployment)
docker build -t inference-worker:latest ./inference_container

# Preprocess
docker build -t preprocess:latest ./preprocess_container

# Training (GRU/LSTM)
docker build -t train:latest ./train_container

# Prophet (non-ML)
docker build -t nonml:latest ./nonml_container

# Eval
docker build -t eval:latest ./eval_container

# MLflow
docker build -t mlflow:latest ./mlflow

# FastAPI Gateway
docker build -t fastapi-app:latest ./fastapi-app
```

### 4. Tag and Push to Artifact Registry

```powershell
$REGISTRY = "europe-west3-docker.pkg.dev/saas-nokia-an-rnd/flts-ml-pipeline"

# Tag images
docker tag inference-http:latest $REGISTRY/inference-http:latest
docker tag inference-worker:latest $REGISTRY/inference-worker:latest
docker tag preprocess:latest $REGISTRY/preprocess:latest
docker tag train:latest $REGISTRY/train:latest
docker tag nonml:latest $REGISTRY/nonml:latest
docker tag eval:latest $REGISTRY/eval:latest
docker tag mlflow:latest $REGISTRY/mlflow:latest
docker tag fastapi-app:latest $REGISTRY/fastapi-app:latest

# Push images
docker push $REGISTRY/inference-http:latest
docker push $REGISTRY/inference-worker:latest
docker push $REGISTRY/preprocess:latest
docker push $REGISTRY/train:latest
docker push $REGISTRY/nonml:latest
docker push $REGISTRY/eval:latest
docker push $REGISTRY/mlflow:latest
docker push $REGISTRY/fastapi-app:latest
```

### 5. Deploy to GKE

```powershell
kubectl apply -k .k8s-gke/
```

### 6. Verify Deployment

```powershell
# Check all pods
kubectl get pods

# Check services (wait for EXTERNAL-IP)
kubectl get svc

# Check HPAs
kubectl get hpa

# Describe a specific HPA
kubectl describe hpa inference-http-hpa
```

### 7. Run the Preprocess Job

The preprocess job triggers the full pipeline (preprocess → train → eval → inference ready):

```powershell
# Delete previous job if exists
kubectl delete job preprocess --ignore-not-found

# Apply the job
kubectl apply -f .k8s-gke/pipeline/preprocess.yaml

# Watch job progress
kubectl logs -f job/preprocess
```

### 8. Run Locust Load Tests

```powershell
# Quick smoke test (30 seconds, 50 users)
kubectl exec deployment/locust-master -- locust --headless --host=http://inference-http:8000 -u 50 -r 10 -t 30s --only-summary

# Full load test (2 minutes, 150 users)
kubectl exec deployment/locust-master -- locust --headless --host=http://inference-http:8000 -u 150 -r 25 -t 120s --only-summary

# Access Locust web UI
kubectl port-forward svc/locust-master 8089:8089
# Open http://localhost:8089 in browser
```

---

## Troubleshooting

### Authentication Issues

**Symptom:** `ERROR: (gcloud.container.clusters.get-credentials) ... Permission denied`

```powershell
# Re-authenticate
gcloud auth login
gcloud auth application-default login

# Verify account
gcloud auth list
```

### Image Push Failures

**Symptom:** `denied: Permission denied` when pushing to Artifact Registry

```powershell
# Re-configure Docker authentication
gcloud auth configure-docker europe-west3-docker.pkg.dev

# Verify repository exists
gcloud artifacts repositories list --location=europe-west3
```

### Pods Stuck in ContainerCreating

**Symptom:** Pods remain in `ContainerCreating` state

```powershell
# Check pod events
kubectl describe pod <pod-name>

# Common causes:
# - Image pull errors → verify image exists in Artifact Registry
# - PVC binding issues → check persistent volume claims
# - Resource constraints → check node capacity
kubectl get events --sort-by='.lastTimestamp'
```

### Service External IP Delays

**Symptom:** Service shows `<pending>` for EXTERNAL-IP

```powershell
# This is normal - GKE LoadBalancer provisioning takes 1-3 minutes
kubectl get svc -w  # Watch until IP appears

# If pending > 5 minutes, check events
kubectl describe svc <service-name>
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| View pods | `kubectl get pods` |
| View logs | `kubectl logs -f deployment/inference-http` |
| Shell into pod | `kubectl exec -it deployment/inference-http -- /bin/sh` |
| Check HPA | `kubectl get hpa` |
| Port-forward MLflow | `kubectl port-forward svc/mlflow 5000:5000` |
| Port-forward MinIO | `kubectl port-forward svc/minio 9001:9001` |
| Restart deployment | `kubectl rollout restart deployment/inference-http` |
