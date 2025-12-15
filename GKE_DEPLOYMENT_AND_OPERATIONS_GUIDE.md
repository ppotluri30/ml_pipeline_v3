# FLTS ML Pipeline: GKE Deployment and Operations Guide

**Document Version:** 1.0  
**Date:** January 2025  
**Platform:** Google Kubernetes Engine (GKE)  
**Pipeline:** FLTS Custom ML Pipeline (NOT Kubeflow)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [GCP Prerequisites](#2-gcp-prerequisites)
3. [Building and Pushing Docker Images](#3-building-and-pushing-docker-images)
4. [Creating the GKE Cluster](#4-creating-the-gke-cluster)
5. [Deploying All Required Microservices](#5-deploying-all-required-microservices)
6. [Running the Training Pipeline](#6-running-the-training-pipeline)
7. [Running the Inference Pipeline](#7-running-the-inference-pipeline)
8. [Running Locust Load Testing](#8-running-locust-load-testing)
9. [Operations & SRE Runbook](#9-operations--sre-runbook)
10. [Final End-to-End Checklist](#10-final-end-to-end-checklist)

---

## 1. Project Overview

### 1.1 What This Pipeline Is

The **FLTS ML Pipeline** is a **custom, event-driven machine learning system** designed for time-series forecasting. It is a purpose-built microservices architecture that runs on Kubernetes—**this is NOT Kubeflow, Vertex AI, or any managed ML platform**.

### 1.2 Architecture Summary

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         FLTS ML PIPELINE ON GKE                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│   │  preprocess │───▶│    Kafka    │───▶│  train-gru  │                 │
│   │    (Job)    │    │   (9092)    │    │  train-lstm │                 │
│   └─────────────┘    │             │    │ nonml-prophet│                 │
│          │           └──────┬──────┘    └──────┬──────┘                 │
│          │                  │                  │                        │
│          ▼                  │                  ▼                        │
│   ┌─────────────┐           │           ┌─────────────┐                 │
│   │   MinIO     │◀──────────┘           │    eval     │                 │
│   │  (S3-compat)│                       │ (promoter)  │                 │
│   └──────┬──────┘                       └──────┬──────┘                 │
│          │                                     │                        │
│          ▼                                     ▼                        │
│   ┌─────────────┐                       ┌─────────────┐                 │
│   │   MLflow    │◀──────────────────────│model-selected│                │
│   │  (tracking) │                       │   (topic)   │                 │
│   └─────────────┘                       └──────┬──────┘                 │
│                                                │                        │
│   ┌─────────────┐    ┌─────────────┐          │                        │
│   │fastapi-app  │◀───│inference-http│◀─────────┘                        │
│   │ (gateway)   │    │  (REST API) │    ┌─────────────┐                 │
│   └─────────────┘    └─────────────┘    │inference-   │                 │
│                             ▲           │  worker     │                 │
│                             │           │(Kafka batch)│                 │
│                      ┌──────┴──────┐    └─────────────┘                 │
│                      │   Locust    │                                    │
│                      │ (load test) │                                    │
│                      └─────────────┘                                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Component Inventory

| Component | Purpose | Replicas | Exposed |
|-----------|---------|----------|---------|
| **Kafka** | Event streaming (KRaft mode, no Zookeeper) | 1 StatefulSet | Internal |
| **MinIO** | S3-compatible object storage | 1 Deployment | Console: LoadBalancer |
| **PostgreSQL** | MLflow backend database | 1 StatefulSet | Internal |
| **MLflow** | Experiment tracking, model registry | 1 Deployment | LoadBalancer |
| **FastAPI-app** | MinIO gateway proxy | 1 Deployment | Internal |
| **preprocess** | Data preprocessing job | Job (on-demand) | N/A |
| **train-gru** | GRU model trainer | 1 Deployment | Internal |
| **train-lstm** | LSTM model trainer | 1 Deployment | Internal |
| **nonml-prophet** | Prophet model trainer | 1 Deployment | Internal |
| **eval** | Model evaluation & promotion | 1 Deployment | Internal |
| **inference-http** | REST API for predictions | 3-10 (HPA) | LoadBalancer |
| **inference-worker** | Kafka batch inference | 2-5 (HPA) | Internal |
| **locust-master** | Load test orchestrator | 1 Deployment | LoadBalancer |
| **locust-worker** | Load test generators | 4 Deployments | Internal |

### 1.4 Kafka Topics

| Topic | Producer | Consumer(s) |
|-------|----------|-------------|
| `training-data` | preprocess | train-gru, train-lstm, nonml-prophet |
| `inference-data` | preprocess | inference-worker |
| `model-training` | trainers | eval |
| `model-selected` | eval | inference-http, inference-worker |

### 1.5 MinIO Buckets

| Bucket | Purpose |
|--------|---------|
| `dataset` | Raw input CSV data |
| `processed-data` | Parquet files + metadata |
| `mlflow` | MLflow artifact storage |
| `model-promotion` | Promotion pointers (`current.json`) |
| `inference-logs` | Binary inference logs |
| `inference-txt-logs` | JSONL prediction logs |

---

## 2. GCP Prerequisites

### 2.1 Required GCP Resources

| Resource | Value |
|----------|-------|
| **Project ID** | `saas-nokia-an-rnd` |
| **Region** | `europe-west3` (Frankfurt) |
| **GKE Cluster** | `aiml-dev-xhckg-gke-cluster` |
| **Artifact Registry** | `europe-west3-docker.pkg.dev/saas-nokia-an-rnd/aiml-dev-registry` |

### 2.2 Required Permissions

Your GCP account needs:

| Permission | Scope | Why |
|------------|-------|-----|
| `container.clusters.get` | Cluster | kubectl access |
| `container.pods.*` | Namespace | Deploy workloads |
| `artifactregistry.repositories.uploadArtifacts` | Repository | Push images |
| `artifactregistry.repositories.downloadArtifacts` | Repository | Pull images |

**Note:** This deployment does NOT require `cluster-admin` privileges. All autoscaling uses standard CPU-based HPA.

### 2.3 Install Required CLI Tools

#### On Windows (PowerShell)

```powershell
# Install gcloud CLI
# Download from: https://cloud.google.com/sdk/docs/install

# After installation, initialize
gcloud init

# Install kubectl component
gcloud components install kubectl

# Verify installations
gcloud version
kubectl version --client
docker --version
```

#### On macOS/Linux

```bash
# Install gcloud CLI (macOS)
brew install google-cloud-sdk

# Install kubectl
gcloud components install kubectl

# Verify
gcloud version
kubectl version --client
docker --version
```

### 2.4 Authenticate to GCP

```powershell
# Step 1: Login to your Google account
gcloud auth login

# Step 2: Set the project
gcloud config set project saas-nokia-an-rnd

# Step 3: Configure Docker for Artifact Registry
gcloud auth configure-docker europe-west3-docker.pkg.dev

# Step 4: Verify authentication
gcloud auth list
gcloud config list project
```

### 2.5 Connect to the GKE Cluster

```powershell
# Get cluster credentials (one-time setup)
gcloud container clusters get-credentials aiml-dev-xhckg-gke-cluster `
    --region europe-west3 `
    --project saas-nokia-an-rnd

# Verify connection
kubectl cluster-info
kubectl get nodes

# Expected output:
# NAME                                                  STATUS   ROLES    AGE     VERSION
# gke-aiml-dev-xhckg-gke-c-default-pool-xxxxx-xxxx     Ready    <none>   7d      v1.31.x
```

---

## 3. Building and Pushing Docker Images

### 3.1 Image Inventory

| Image Name | Source Directory | Purpose |
|------------|------------------|---------|
| `inference-http` | `./inference_container` | HTTP prediction API |
| `inference-worker` | `./inference_container` | Kafka batch processor |
| `preprocess` | `./preprocess_container` | Data preprocessing |
| `train` | `./train_container` | GRU/LSTM trainer |
| `nonml` | `./nonML_container` | Prophet trainer |
| `eval` | `./eval_container` | Model evaluation |
| `mlflow` | `./mlflow` | MLflow tracking server |
| `fastapi-app` | `./minio` | MinIO gateway |

### 3.2 Build All Images

From the repository root directory (`ml_pipeline_v3`):

```powershell
# Navigate to repository root
cd C:\Users\ppotluri\Desktop\ml_pipeline_v3

# Build all 8 images
Write-Host "Building inference-http..."
docker build -t inference-http:latest ./inference_container

Write-Host "Building inference-worker..."
docker build -t inference-worker:latest ./inference_container

Write-Host "Building preprocess..."
docker build -t preprocess:latest ./preprocess_container

Write-Host "Building train (GRU/LSTM)..."
docker build -t train:latest ./train_container

Write-Host "Building nonml (Prophet)..."
docker build -t nonml:latest ./nonML_container

Write-Host "Building eval..."
docker build -t eval:latest ./eval_container

Write-Host "Building mlflow..."
docker build -t mlflow:latest ./mlflow

Write-Host "Building fastapi-app..."
docker build -t fastapi-app:latest ./minio

# Verify all images built
docker images | Select-String -Pattern "inference|preprocess|train|nonml|eval|mlflow|fastapi"
```

### 3.3 Tag Images for Artifact Registry

```powershell
# Define registry path
$REGISTRY = "europe-west3-docker.pkg.dev/saas-nokia-an-rnd/aiml-dev-registry"
$TAG = "latest"  # Or use a version tag like "v1.0.0"

# Tag all images
docker tag inference-http:latest "${REGISTRY}/inference-http:${TAG}"
docker tag inference-worker:latest "${REGISTRY}/inference-worker:${TAG}"
docker tag preprocess:latest "${REGISTRY}/preprocess:${TAG}"
docker tag train:latest "${REGISTRY}/train:${TAG}"
docker tag nonml:latest "${REGISTRY}/nonml:${TAG}"
docker tag eval:latest "${REGISTRY}/eval:${TAG}"
docker tag mlflow:latest "${REGISTRY}/mlflow:${TAG}"
docker tag fastapi-app:latest "${REGISTRY}/fastapi-app:${TAG}"

# Verify tags
docker images | Select-String $REGISTRY
```

### 3.4 Push Images to Artifact Registry

```powershell
# Ensure Docker auth is configured
gcloud auth configure-docker europe-west3-docker.pkg.dev

# Push all images (this may take several minutes)
$REGISTRY = "europe-west3-docker.pkg.dev/saas-nokia-an-rnd/aiml-dev-registry"

Write-Host "Pushing inference-http..."
docker push "${REGISTRY}/inference-http:latest"

Write-Host "Pushing inference-worker..."
docker push "${REGISTRY}/inference-worker:latest"

Write-Host "Pushing preprocess..."
docker push "${REGISTRY}/preprocess:latest"

Write-Host "Pushing train..."
docker push "${REGISTRY}/train:latest"

Write-Host "Pushing nonml..."
docker push "${REGISTRY}/nonml:latest"

Write-Host "Pushing eval..."
docker push "${REGISTRY}/eval:latest"

Write-Host "Pushing mlflow..."
docker push "${REGISTRY}/mlflow:latest"

Write-Host "Pushing fastapi-app..."
docker push "${REGISTRY}/fastapi-app:latest"

Write-Host "All images pushed successfully!"
```

### 3.5 Verify Images in Artifact Registry

```powershell
# List images in the registry
gcloud artifacts docker images list europe-west3-docker.pkg.dev/saas-nokia-an-rnd/aiml-dev-registry

# Or via GCP Console:
# https://console.cloud.google.com/artifacts/docker/saas-nokia-an-rnd/europe-west3/aiml-dev-registry
```

---

## 4. Creating the GKE Cluster

### 4.1 Existing Cluster

The GKE cluster already exists. If you need to recreate it:

```powershell
# This is for reference only - the cluster already exists
gcloud container clusters create aiml-dev-xhckg-gke-cluster `
    --region europe-west3 `
    --project saas-nokia-an-rnd `
    --machine-type e2-standard-4 `
    --num-nodes 3 `
    --enable-autoscaling `
    --min-nodes 2 `
    --max-nodes 6 `
    --disk-size 100 `
    --release-channel regular
```

### 4.2 Cluster Specifications

| Setting | Value |
|---------|-------|
| Name | `aiml-dev-xhckg-gke-cluster` |
| Region | `europe-west3` |
| Node Pool | Default |
| Machine Type | e2-standard-4 (4 vCPU, 16 GB RAM) |
| Storage Class | `standard-rwo` (GKE standard) |
| Kubernetes Version | 1.31.x |

### 4.3 Storage Configuration

The pipeline uses GKE's default `standard-rwo` StorageClass for:

- **PostgreSQL**: 5Gi PVC for MLflow database
- **MinIO**: 10Gi PVC for object storage

```powershell
# Verify storage class exists
kubectl get storageclass

# Expected output:
# NAME                     PROVISIONER             RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION
# standard-rwo (default)   pd.csi.storage.gke.io   Delete          WaitForFirstConsumer   true
```

---

## 5. Deploying All Required Microservices

### 5.1 Deployment Method: Kustomize

The `.k8s-gke/` directory contains all Kubernetes manifests organized with Kustomize:

```
.k8s-gke/
├── kustomization.yaml     # Main Kustomize config
├── backend/               # Infrastructure services
│   ├── kafka.yaml
│   ├── minio.yaml
│   ├── minio-init-job.yaml
│   ├── mlflow.yaml
│   └── fastapi-app.yaml
├── autoscaling/           # HPA configurations
│   ├── hpa-inference-http.yaml
│   └── hpa-inference-worker.yaml
├── pipeline/              # ML pipeline services
│   ├── preprocess.yaml
│   ├── training.yaml
│   └── eval.yaml
├── inference/             # Inference services
│   ├── deployment.yaml
│   └── service.yaml
├── worker/                # Kafka consumer worker
│   └── deployment.yaml
└── locust/                # Load testing
    ├── configmap.yaml
    ├── master.yaml
    └── worker.yaml
```

### 5.2 Deploy Everything in One Command

```powershell
# Ensure you're connected to the cluster
kubectl cluster-info

# Deploy the entire stack
kubectl apply -k .k8s-gke/

# Expected output:
# configmap/preprocess-config created
# configmap/locust-scripts created
# secret/mlflow-postgres-secret created
# persistentvolumeclaim/minio-data created
# service/kafka-headless created
# service/kafka created
# service/minio created
# service/minio-console created
# service/mlflow-postgres created
# service/mlflow created
# service/fastapi-app created
# service/inference-http created
# service/eval created
# service/locust-master created
# deployment.apps/minio created
# deployment.apps/mlflow created
# deployment.apps/fastapi-app created
# deployment.apps/inference-http created
# deployment.apps/inference-worker created
# deployment.apps/train-gru created
# deployment.apps/train-lstm created
# deployment.apps/nonml-prophet created
# deployment.apps/eval created
# deployment.apps/locust-master created
# deployment.apps/locust-worker created
# statefulset.apps/kafka created
# statefulset.apps/mlflow-postgres created
# job.batch/minio-init created
# job.batch/preprocess created
# horizontalpodautoscaler.autoscaling/inference-http-hpa created
# horizontalpodautoscaler.autoscaling/inference-worker-hpa created
```

### 5.3 Verify Deployment Status

```powershell
# Wait for all pods to be running
kubectl get pods -w

# Check specific component groups
kubectl get pods -l tier=training
kubectl get pods -l tier=serving
kubectl get pods -l app=locust

# Check all services
kubectl get svc

# Check HPAs
kubectl get hpa
```

### 5.4 Expected Pod Status (Healthy)

```
NAME                               READY   STATUS      RESTARTS   AGE
kafka-0                            1/1     Running     0          5m
minio-xxxxxxxx-xxxxx              1/1     Running     0          5m
minio-init-xxxxx                  0/1     Completed   0          4m
mlflow-postgres-0                  1/1     Running     0          5m
mlflow-xxxxxxxx-xxxxx             1/1     Running     0          4m
fastapi-app-xxxxxxxx-xxxxx        1/1     Running     0          4m
train-gru-xxxxxxxx-xxxxx          1/1     Running     0          4m
train-lstm-xxxxxxxx-xxxxx         1/1     Running     0          4m
nonml-prophet-xxxxxxxx-xxxxx      1/1     Running     0          4m
eval-xxxxxxxx-xxxxx               1/1     Running     0          4m
inference-http-xxxxxxxx-xxxxx     1/1     Running     0          3m
inference-http-xxxxxxxx-xxxxx     1/1     Running     0          3m
inference-http-xxxxxxxx-xxxxx     1/1     Running     0          3m
inference-worker-xxxxxxxx-xxxxx   1/1     Running     0          3m
inference-worker-xxxxxxxx-xxxxx   1/1     Running     0          3m
locust-master-xxxxxxxx-xxxxx      1/1     Running     0          3m
locust-worker-xxxxxxxx-xxxxx      1/1     Running     0          3m
locust-worker-xxxxxxxx-xxxxx      1/1     Running     0          3m
locust-worker-xxxxxxxx-xxxxx      1/1     Running     0          3m
locust-worker-xxxxxxxx-xxxxx      1/1     Running     0          3m
```

### 5.5 Get External IPs

GKE provisions LoadBalancer IPs for external services:

```powershell
# Wait for EXTERNAL-IP to be assigned (may take 1-2 minutes)
kubectl get svc -w

# Get specific service IPs
$INFERENCE_IP = kubectl get svc inference-http -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
$MLFLOW_IP = kubectl get svc mlflow -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
$MINIO_CONSOLE_IP = kubectl get svc minio-console -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
$LOCUST_IP = kubectl get svc locust-master -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

Write-Host "Inference API:   http://${INFERENCE_IP}:8000"
Write-Host "MLflow UI:       http://${MLFLOW_IP}:5000"
Write-Host "MinIO Console:   http://${MINIO_CONSOLE_IP}:9001"
Write-Host "Locust UI:       http://${LOCUST_IP}:8089"
```

### 5.6 Component Resource Allocations

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| Kafka | 250m | 1000m | 512Mi | 1Gi |
| MinIO | 100m | 500m | 256Mi | 512Mi |
| PostgreSQL | 100m | 500m | 256Mi | 512Mi |
| MLflow | (default) | (default) | (default) | (default) |
| FastAPI-app | 100m | 500m | 128Mi | 256Mi |
| train-* | 200m | 1000m | 512Mi | 2Gi |
| eval | 100m | 500m | 256Mi | 512Mi |
| inference-http | 500m | 2000m | 768Mi | 1500Mi |
| inference-worker | 500m | 2000m | 768Mi | 1500Mi |
| locust-master | 500m | 1000m | 512Mi | 1Gi |
| locust-worker | 250m | 500m | 256Mi | 512Mi |

---

## 6. Running the Training Pipeline

### 6.1 Pipeline Flow

```
preprocess → Kafka (training-data topic) → trainers → Kafka (model-training) → eval → promotion
```

### 6.2 Upload Dataset to MinIO

Before running the pipeline, upload your dataset:

```powershell
# Port-forward to MinIO console
kubectl port-forward svc/minio 9001:9001

# Access MinIO Console at http://localhost:9001
# Login: minioadmin / minioadmin
# Upload PobleSec.csv to the "dataset" bucket
```

Or via CLI:

```powershell
# Install MinIO client
# Download from: https://dl.min.io/client/mc/release/windows-amd64/mc.exe

# Configure alias (using port-forward)
kubectl port-forward svc/minio 9000:9000
mc alias set gke-minio http://localhost:9000 minioadmin minioadmin

# Upload dataset
mc cp ./dataset/PobleSec.csv gke-minio/dataset/
mc ls gke-minio/dataset/
```

### 6.3 Run the Preprocess Job

```powershell
# Delete any previous preprocess job
kubectl delete job preprocess --ignore-not-found

# Apply the preprocess job
kubectl apply -f .k8s-gke/pipeline/preprocess.yaml

# Watch the job logs
kubectl logs -f job/preprocess

# Expected log output:
# [preprocess] Loading PobleSec.csv from bucket dataset
# [preprocess] Applying transformations...
# [preprocess] Writing processed data to processed-data bucket
# [preprocess] Publishing to training-data topic
# [preprocess] Config hash: abc123...
# [preprocess] SUCCESS - preprocessing complete
```

### 6.4 Monitor Training

```powershell
# Watch all trainer logs in separate terminals
kubectl logs -f deployment/train-gru
kubectl logs -f deployment/train-lstm
kubectl logs -f deployment/nonml-prophet

# Or combined view
kubectl logs -l tier=training -f --max-log-requests=5
```

Training success indicators:

```
# train-gru output
[train-gru] Received training-data message: config_hash=abc123
[train-gru] Training GRU model...
[train-gru] Epoch 1/3 - loss: 0.0123
[train-gru] Epoch 2/3 - loss: 0.0089
[train-gru] Epoch 3/3 - loss: 0.0067
[train-gru] Logging to MLflow run_id=xxxxx
[train-gru] Publishing to model-training topic
[train-gru] train_success_publish: GRU completed
```

### 6.5 Monitor Evaluation and Promotion

```powershell
# Watch eval logs
kubectl logs -f deployment/eval

# Expected output
[eval] Waiting for all model types: GRU, LSTM, PROPHET
[eval] Received model-training: GRU (config_hash=abc123)
[eval] Received model-training: LSTM (config_hash=abc123)
[eval] Received model-training: PROPHET (config_hash=abc123)
[eval] All models received for config_hash=abc123
[eval] promotion_scoreboard: GRU=0.042, LSTM=0.045, PROPHET=0.068
[eval] Best model: GRU (score=0.042)
[eval] Writing promotion pointer: model-promotion/current.json
[eval] promotion_artifacts_ok: run_id=xxxxx, model_type=GRU
[eval] Publishing to model-selected topic
```

### 6.6 Verify Promotion

```powershell
# Check that promotion pointer exists
kubectl exec deployment/fastapi-app -- wget -qO- http://localhost:8000/download/model-promotion/current.json

# Expected output:
# {
#   "model_uri": "runs:/xxxxx/GRU",
#   "run_id": "xxxxx",
#   "model_type": "GRU",
#   "config_hash": "abc123...",
#   "score": 0.042,
#   "rmse": 1.23,
#   "promoted_at": "2025-01-15T10:30:00Z"
# }
```

### 6.7 View Training Runs in MLflow

```powershell
# Get MLflow external IP
kubectl get svc mlflow

# Access MLflow UI
# http://<EXTERNAL-IP>:5000
```

---

## 7. Running the Inference Pipeline

### 7.1 Inference Architecture

The inference system has two components:

1. **inference-http**: REST API for synchronous predictions (LoadBalancer exposed)
2. **inference-worker**: Kafka consumer for batch/async inference (internal)

### 7.2 Verify Inference Readiness

```powershell
# Check inference-http pods are ready
kubectl get pods -l app=inference-http

# Check readiness probe
kubectl exec deployment/inference-http -- curl -s http://localhost:8000/internal/ready

# Expected: {"status": "ready", "model_loaded": true}

# Check health endpoint
kubectl exec deployment/inference-http -- curl -s http://localhost:8000/healthz

# Expected: {"status": "healthy"}
```

### 7.3 Make a Prediction Request

```powershell
# Get inference external IP
$INFERENCE_IP = kubectl get svc inference-http -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Create a test payload
$payload = @{
    index_col = "ts"
    inference_length = 1
    data = @{
        ts = @("2025-01-15T10:00:00", "2025-01-15T10:01:00", "2025-01-15T10:02:00")
        down = @(5000000.0, 5100000.0, 5200000.0)
        up = @(1000.0, 1100.0, 1200.0)
        rnti_count = @(2000.0, 2010.0, 2020.0)
        mcs_down = @(10.0, 10.5, 11.0)
        mcs_down_var = @(50.0, 51.0, 52.0)
        mcs_up = @(12.0, 12.5, 13.0)
        mcs_up_var = @(40.0, 41.0, 42.0)
        rb_down = @(0.05, 0.051, 0.052)
        rb_down_var = @(1e-7, 1.1e-7, 1.2e-7)
        rb_up = @(0.01, 0.011, 0.012)
        rb_up_var = @(5e-8, 5.5e-8, 6e-8)
    }
} | ConvertTo-Json -Depth 5

# Make prediction
Invoke-RestMethod -Uri "http://${INFERENCE_IP}:8000/predict" -Method Post -Body $payload -ContentType "application/json"
```

### 7.4 Test Prediction via kubectl

```powershell
# Quick test from inside the cluster
kubectl run test-curl --rm -i --restart=Never --image=curlimages/curl:8.10.1 -- `
    curl -s -X POST http://inference-http:8000/predict `
    -H "Content-Type: application/json" `
    -d '{"index_col":"ts","inference_length":1,"data":{"ts":["2025-01-15T10:00:00"],"down":[5000000],"up":[1000],"rnti_count":[2000],"mcs_down":[10],"mcs_down_var":[50],"mcs_up":[12],"mcs_up_var":[40],"rb_down":[0.05],"rb_down_var":[1e-7],"rb_up":[0.01],"rb_up_var":[5e-8]}}'
```

### 7.5 Check Inference Metrics

```powershell
# Get Prometheus metrics from inference
kubectl exec deployment/inference-http -- curl -s http://localhost:8000/prometheus

# Key metrics:
# inference_requests_total - Total prediction requests
# inference_latency_seconds - Request latency histogram
# inference_queue_len - Current request queue length
# inference_model_type - Currently loaded model
```

### 7.6 HPA Autoscaling Status

```powershell
# Check HPA status
kubectl get hpa

# Expected output:
# NAME                     REFERENCE                    TARGETS   MINPODS   MAXPODS   REPLICAS   AGE
# inference-http-hpa       Deployment/inference-http    35%/60%   3         10        3          1h
# inference-worker-hpa     Deployment/inference-worker  25%/60%   2         5         2          1h

# Detailed HPA info
kubectl describe hpa inference-http-hpa
```

---

## 8. Running Locust Load Testing

### 8.1 Access Locust UI

```powershell
# Get Locust external IP
$LOCUST_IP = kubectl get svc locust-master -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

Write-Host "Locust UI: http://${LOCUST_IP}:8089"
```

Open the URL in a browser and configure:

- **Number of users**: 50-200 (depending on test)
- **Spawn rate**: 10-25 users/second
- **Host**: `http://inference-http:8000` (pre-configured)

### 8.2 Run Headless Load Test

```powershell
# Quick smoke test (30 seconds, 50 users)
kubectl exec deployment/locust-master -- locust --headless `
    --host=http://inference-http:8000 `
    -u 50 -r 10 -t 30s --only-summary

# Medium load test (2 minutes, 150 users)
kubectl exec deployment/locust-master -- locust --headless `
    --host=http://inference-http:8000 `
    -u 150 -r 25 -t 120s --only-summary

# Heavy load test (5 minutes, 300 users)
kubectl exec deployment/locust-master -- locust --headless `
    --host=http://inference-http:8000 `
    -u 300 -r 50 -t 300s --only-summary
```

### 8.3 Expected Load Test Results

For a healthy system with 3 inference-http replicas:

| Metric | Acceptable Range |
|--------|------------------|
| Median Response Time | < 500ms |
| P95 Response Time | < 2000ms |
| P99 Response Time | < 5000ms |
| Failure Rate | < 1% |
| Requests/sec | 50-200 RPS |

### 8.4 Monitor HPA Scaling During Load

```powershell
# Watch HPA in real-time
kubectl get hpa -w

# Watch pod scaling
kubectl get pods -l app=inference-http -w

# Monitor CPU usage
kubectl top pods -l app=inference-http
```

### 8.5 Scale Locust Workers

```powershell
# Scale up for heavier load tests
kubectl scale deployment/locust-worker --replicas=8

# Scale back down
kubectl scale deployment/locust-worker --replicas=4
```

### 8.6 Warmup Configuration

The Locust script includes automatic warmup to prevent cold-start latency spikes:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LOCUST_WARMUP_REQUESTS` | 10 | Warmup requests before test |
| `LOCUST_WARMUP_TIMEOUT` | 30 | Timeout per warmup request |
| `LOCUST_WARMUP_DISABLE` | 0 | Set to 1 to disable warmup |
| `DEBUG_LOCUST_WARMUP` | 0 | Set to 1 for verbose warmup logs |

---

## 9. Operations & SRE Runbook

### 9.1 Health Check Commands

```powershell
# All pods status
kubectl get pods

# Pods with issues
kubectl get pods --field-selector=status.phase!=Running

# Recent events (errors)
kubectl get events --sort-by='.lastTimestamp' | Select-Object -Last 20

# Describe problematic pod
kubectl describe pod <pod-name>

# Container logs
kubectl logs <pod-name> -c <container-name> --tail=100
```

### 9.2 Common Issues and Resolutions

#### Issue: Pod in CrashLoopBackOff

```powershell
# Get pod status
kubectl describe pod <pod-name>

# Check logs from previous crash
kubectl logs <pod-name> --previous

# Common causes:
# - OOMKilled: Increase memory limits
# - Image pull failed: Check Artifact Registry permissions
# - Dependency not ready: Check init containers
```

**Resolution:**

```powershell
# For OOM issues
kubectl patch deployment <deployment-name> -p '{"spec":{"template":{"spec":{"containers":[{"name":"<container>","resources":{"limits":{"memory":"2Gi"}}}]}}}}'

# Restart deployment
kubectl rollout restart deployment/<deployment-name>
```

#### Issue: HPA Not Scaling

```powershell
# Check HPA status
kubectl describe hpa inference-http-hpa

# Check if metrics-server is running
kubectl get deployment metrics-server -n kube-system

# Check resource metrics
kubectl top pods -l app=inference-http
```

**Resolution:**

```powershell
# If metrics unavailable, restart metrics-server
kubectl rollout restart deployment/metrics-server -n kube-system

# Verify HPA is targeting correct deployment
kubectl get hpa inference-http-hpa -o yaml
```

#### Issue: Kafka Connection Failures

```powershell
# Check Kafka pod
kubectl logs kafka-0

# Test Kafka connectivity
kubectl run kafka-test --rm -i --restart=Never --image=apache/kafka:3.9.1 -- /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --list
```

**Resolution:**

```powershell
# Restart Kafka StatefulSet
kubectl rollout restart statefulset/kafka

# Wait for ready
kubectl wait --for=condition=ready pod/kafka-0 --timeout=120s
```

#### Issue: Model Not Loading in Inference

```powershell
# Check inference logs
kubectl logs -l app=inference-http --tail=100 | Select-String "scaler|model|promotion"

# Verify promotion pointer exists
kubectl exec deployment/fastapi-app -- wget -qO- http://localhost:8000/download/model-promotion/current.json

# Check MLflow connectivity
kubectl exec deployment/inference-http -- curl -s http://mlflow:5000/api/2.0/mlflow/experiments/list
```

**Resolution:**

```powershell
# Re-run eval to create promotion pointer
kubectl delete job preprocess --ignore-not-found
kubectl apply -f .k8s-gke/pipeline/preprocess.yaml

# Or manually trigger eval
kubectl rollout restart deployment/eval
```

### 9.3 Log Aggregation

```powershell
# All training logs
kubectl logs -l tier=training -f --max-log-requests=5

# All inference logs
kubectl logs -l tier=serving -f --max-log-requests=5

# Save logs to file
kubectl logs deployment/inference-http --since=1h > inference-logs.txt

# Export logs for analysis
kubectl logs -l app=inference-http --since=30m --timestamps > inference-analysis.log
```

### 9.4 Scaling Operations

```powershell
# Manual scale up
kubectl scale deployment/inference-http --replicas=5

# Reset to HPA-managed
kubectl scale deployment/inference-http --replicas=3

# Temporarily disable HPA
kubectl delete hpa inference-http-hpa
kubectl scale deployment/inference-http --replicas=5

# Re-enable HPA
kubectl apply -f .k8s-gke/autoscaling/hpa-inference-http.yaml
```

### 9.5 Rolling Updates

```powershell
# Update image
$NEW_TAG = "v1.1.0"
kubectl set image deployment/inference-http inference-http=europe-west3-docker.pkg.dev/saas-nokia-an-rnd/aiml-dev-registry/inference-http:$NEW_TAG

# Watch rollout
kubectl rollout status deployment/inference-http

# Rollback if needed
kubectl rollout undo deployment/inference-http

# Check rollout history
kubectl rollout history deployment/inference-http
```

### 9.6 Backup and Recovery

```powershell
# Export all resources
kubectl get all -o yaml > gke-backup.yaml

# Backup PVC data (MinIO)
kubectl exec minio-xxxxxxxx-xxxxx -- tar -cvf /tmp/minio-backup.tar /data
kubectl cp default/minio-xxxxxxxx-xxxxx:/tmp/minio-backup.tar ./minio-backup.tar

# Backup PostgreSQL (MLflow)
kubectl exec mlflow-postgres-0 -- pg_dump -U mlflow mlflow > mlflow-backup.sql
```

### 9.7 Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Pod CPU | > 70% | > 90% |
| Pod Memory | > 80% | > 95% |
| P95 Latency | > 2s | > 5s |
| Error Rate | > 1% | > 5% |
| Kafka Lag | > 100 | > 1000 |
| Pod Restarts | > 3/hr | > 10/hr |

### 9.8 Emergency Procedures

#### Complete Pipeline Restart

```powershell
# Delete all workloads
kubectl delete deployment --all
kubectl delete statefulset --all
kubectl delete job --all

# Re-deploy
kubectl apply -k .k8s-gke/
```

#### Drain and Cordon Node

```powershell
# Cordon node (no new pods)
kubectl cordon <node-name>

# Drain node (evict pods)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Uncordon after maintenance
kubectl uncordon <node-name>
```

---

## 10. Final End-to-End Checklist

### 10.1 Pre-Deployment Checklist

| Step | Command | Expected Result |
|------|---------|-----------------|
| ☐ GCP Auth | `gcloud auth list` | Account listed as ACTIVE |
| ☐ Project Set | `gcloud config get-value project` | `saas-nokia-an-rnd` |
| ☐ Docker Auth | `gcloud auth configure-docker europe-west3-docker.pkg.dev` | Configured |
| ☐ Cluster Connected | `kubectl get nodes` | Nodes listed as Ready |
| ☐ Images Built | `docker images \| Select-String inference` | 8 images listed |
| ☐ Images Pushed | `gcloud artifacts docker images list ...` | Images in registry |

### 10.2 Deployment Verification Checklist

| Step | Command | Expected Result |
|------|---------|-----------------|
| ☐ Apply Manifests | `kubectl apply -k .k8s-gke/` | All resources created |
| ☐ Backend Running | `kubectl get pods -l app=kafka` | 1/1 Running |
| ☐ MinIO Ready | `kubectl get pods -l app=minio` | 1/1 Running |
| ☐ MinIO Init Done | `kubectl get job minio-init` | 1/1 Completed |
| ☐ MLflow Running | `kubectl get pods -l app=mlflow` | 1/1 Running |
| ☐ Trainers Running | `kubectl get pods -l tier=training` | 3 pods Running |
| ☐ Eval Running | `kubectl get pods -l app=eval` | 1/1 Running |
| ☐ Inference Running | `kubectl get pods -l app=inference-http` | 3/3 Running |
| ☐ HPAs Active | `kubectl get hpa` | 2 HPAs, targets showing |
| ☐ LoadBalancers Ready | `kubectl get svc` | EXTERNAL-IPs assigned |

### 10.3 Pipeline Execution Checklist

| Step | Command | Expected Result |
|------|---------|-----------------|
| ☐ Dataset Uploaded | `mc ls gke-minio/dataset/` | PobleSec.csv present |
| ☐ Preprocess Run | `kubectl logs job/preprocess` | SUCCESS message |
| ☐ Training Started | `kubectl logs -l tier=training --tail=10` | Training logs appear |
| ☐ Models in MLflow | Access MLflow UI | Runs visible |
| ☐ Eval Promotion | `kubectl logs deployment/eval --tail=20` | promotion_artifacts_ok |
| ☐ Pointer Created | Check current.json | Model URI present |
| ☐ Inference Ready | `curl /internal/ready` | model_loaded: true |

### 10.4 Load Testing Checklist

| Step | Command | Expected Result |
|------|---------|-----------------|
| ☐ Locust UI Access | Open http://LOCUST_IP:8089 | UI loads |
| ☐ Workers Connected | Check Locust UI | 4 workers connected |
| ☐ Smoke Test Pass | 50 users, 30s | 0% failure rate |
| ☐ Medium Load Pass | 150 users, 2min | <1% failure, P95<2s |
| ☐ HPA Triggered | Watch during load | Replicas increase |

### 10.5 Production Readiness Checklist

| Category | Item | Status |
|----------|------|--------|
| **Security** | ☐ No default passwords in production | |
| **Security** | ☐ Network policies configured | |
| **Availability** | ☐ minReplicas >= 2 for critical services | ✓ |
| **Availability** | ☐ PodDisruptionBudgets configured | |
| **Monitoring** | ☐ Prometheus scraping enabled | ✓ |
| **Monitoring** | ☐ Alerting rules configured | |
| **Backup** | ☐ PVC backup strategy defined | |
| **Backup** | ☐ MLflow database backup scheduled | |
| **Scaling** | ☐ HPA tested under load | ✓ |
| **Scaling** | ☐ Cluster autoscaler enabled | |
| **Documentation** | ☐ Runbook accessible to on-call | ✓ |

---

## Appendix A: Quick Reference Commands

```powershell
# === AUTHENTICATION ===
gcloud auth login
gcloud auth configure-docker europe-west3-docker.pkg.dev
gcloud container clusters get-credentials aiml-dev-xhckg-gke-cluster --region europe-west3

# === DEPLOYMENT ===
kubectl apply -k .k8s-gke/
kubectl delete -k .k8s-gke/

# === STATUS ===
kubectl get pods
kubectl get svc
kubectl get hpa
kubectl get pvc
kubectl top pods

# === LOGS ===
kubectl logs deployment/inference-http -f
kubectl logs -l tier=training -f --max-log-requests=5
kubectl logs job/preprocess

# === DEBUGGING ===
kubectl describe pod <pod-name>
kubectl exec -it deployment/inference-http -- /bin/sh
kubectl port-forward svc/mlflow 5000:5000

# === SCALING ===
kubectl scale deployment/inference-http --replicas=5
kubectl scale deployment/locust-worker --replicas=8

# === UPDATES ===
kubectl set image deployment/inference-http inference-http=$REGISTRY/inference-http:v1.1.0
kubectl rollout status deployment/inference-http
kubectl rollout undo deployment/inference-http

# === CLEANUP ===
kubectl delete job preprocess --ignore-not-found
kubectl delete pod <pod-name> --grace-period=0 --force
```

---

## Appendix B: Environment Variables Reference

### Inference Services

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | minioadmin | MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | minioadmin | MinIO secret key |
| `GATEWAY_URL` | http://fastapi-app:8000 | MinIO gateway URL |
| `MLFLOW_TRACKING_URI` | http://mlflow:5000 | MLflow server URL |
| `MLFLOW_S3_ENDPOINT_URL` | http://minio:9000 | MinIO S3 endpoint |
| `IDENTIFIER` | default | Pipeline identifier |
| `UVICORN_WORKERS` | 2 | HTTP server workers |

### Training Services

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TYPE` | GRU/LSTM/PROPHET | Model architecture |
| `EPOCHS` | 3 | Training epochs |
| `BATCH_SIZE` | 64 | Training batch size |
| `CONSUMER_GROUP_ID` | train-{type} | Kafka consumer group |
| `SKIP_DUPLICATE_CONFIGS` | 1 | Skip duplicate training |

### Locust

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCUST_PAYLOAD_ROWS` | 30 | Rows per request |
| `LOCUST_WARMUP_REQUESTS` | 10 | Warmup request count |
| `PREDICT_USER_WAIT_MIN` | 1 | Min wait between requests |
| `PREDICT_USER_WAIT_MAX` | 2 | Max wait between requests |

---

## Appendix C: Service URLs

| Service | Internal URL | External Access |
|---------|--------------|-----------------|
| Kafka | kafka:9092 | kubectl port-forward |
| MinIO S3 | minio:9000 | kubectl port-forward |
| MinIO Console | minio:9001 | LoadBalancer :9001 |
| MLflow | mlflow:5000 | LoadBalancer :5000 |
| FastAPI Gateway | fastapi-app:8000 | Internal only |
| Inference HTTP | inference-http:8000 | LoadBalancer :8000 |
| Eval | eval:8050 | Internal only |
| Locust Master | locust-master:8089 | LoadBalancer :8089 |

---

**Document End**

*Last Updated: January 2025*  
*Maintainer: FLTS ML Pipeline Team*
