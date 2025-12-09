# FLTS ML Pipeline - Helm Chart

Kubernetes deployment for the FLTS time-series ML training and inference platform.

## Overview

This Helm chart deploys the complete FLTS pipeline including:
- **Infrastructure**: Kafka, MinIO (S3), PostgreSQL, MLflow
- **Data Pipeline**: Preprocessing, Training (GRU/LSTM/Prophet), Evaluation
- **Inference**: Scalable inference API with HAProxy load balancer
- **Monitoring**: Prometheus + Grafana
- **Load Testing**: Locust (master + workers)

## Prerequisites

- Kubernetes cluster (v1.24+)
- Helm 3.8+
- kubectl configured
- Storage provisioner for PersistentVolumes (or use hostPath for local testing)

### Local Testing

For local development/testing:
```bash
# Option 1: Minikube
minikube start --cpus=4 --memory=8192 --disk-size=50g
minikube addons enable storage-provisioner
minikube addons enable metrics-server  # For HPA

# Option 2: Kind
kind create cluster --config kind-config.yaml
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml
```

## Quick Start

### 1. Install with default values (development)

```bash
cd .helm
helm install flts-pipeline . -f values-dev.yaml
```

### 2. Monitor deployment

```bash
# Watch pods come up
kubectl get pods -w

# Check services
kubectl get svc

# View logs for specific service
kubectl logs -f deployment/inference
```

### 3. Access services

```bash
# Port-forward to access UIs locally
kubectl port-forward svc/mlflow 5000:5000
kubectl port-forward svc/grafana 3000:3000
kubectl port-forward svc/locust 8089:8089
kubectl port-forward svc/inference-lb 8023:80

# Then open in browser:
# MLflow:  http://localhost:5000
# Grafana: http://localhost:3000
# Locust:  http://localhost:8089
# Inference: http://localhost:8023/predict
```

### 4. Run the ML pipeline

```bash
# Trigger preprocessing (creates claim-check messages)
kubectl exec -it deployment/preprocess -- python main.py

# Training happens automatically via Kafka consumers
# Monitor training in MLflow UI

# Check inference is ready
kubectl exec -it deployment/inference -- curl http://localhost:8000/ready

# Run load test via Locust UI or headless
```

## Configuration

### Values Files

Three values files are provided:

1. **values-complete.yaml** - Complete reference with all options documented
2. **values-dev.yaml** - Development environment (low resources, quick iteration)
3. **values-prod.yaml** - Production environment (HA, autoscaling, security hardening)

### Common Overrides

```bash
# Custom identifier
helm install flts-pipeline . --set global.identifier=experiment-42

# Scale inference replicas
helm install flts-pipeline . --set inference.replicas=8

# Enable HPA for inference
helm install flts-pipeline . --set inference.autoscaling.enabled=true

# Use custom image registry
helm install flts-pipeline . --set global.imageRegistry=myregistry.io/flts

# Enable ingress
helm install flts-pipeline . --set ingress.enabled=true
```

### Environment-Specific Deployment

```bash
# Development
helm install flts-dev . -f values-dev.yaml

# Production
helm install flts-prod . -f values-prod.yaml \
  --set minio.auth.existingSecret=minio-prod-creds \
  --set postgres.auth.existingSecret=postgres-prod-creds
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Ingress (optional)                      │
│  mlflow.example.com  grafana.example.com  api.example.com     │
└───────────┬─────────────────┬─────────────────┬────────────────┘
            │                 │                 │
            v                 v                 v
      ┌─────────┐       ┌─────────┐     ┌─────────────┐
      │ MLflow  │       │ Grafana │     │ Inference-LB│
      │  :5000  │       │  :3000  │     │  (HAProxy)  │
      └────┬────┘       └────┬────┘     └──────┬──────┘
           │                 │                  │
           │            ┌────┴────┐        ┌───┴────┐
           │            │Prometheus│        │Inference│
           │            │  :9090   │        │ Pods   │
           │            └─────────┘        │ (1-20)  │
           │                               └────┬────┘
           v                                    │
      ┌─────────┐                               │
      │Postgres │◄──────────────────────────────┤
      │  :5432  │                               │
      └─────────┘                               │
                                                │
      ┌─────────────────────────────────────────┼──────────┐
      │                   Kafka :9092           │          │
      │  Topics: training-data, model-training, │          │
      │          inference-data, model-selected │          │
      └─────────────────┬───────────────────────┘          │
                        │                                  │
         ┌──────────────┼──────────────┬───────────────────┤
         │              │              │                   │
         v              v              v                   v
   ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐
   │Preprocess│  │Train-GRU │  │Train-LSTM│       │   Eval   │
   └────┬─────┘  └────┬─────┘  └────┬─────┘       └────┬─────┘
        │             │              │                  │
        │             │              │                  │
        └─────────────┴──────────────┴──────────────────┘
                            │
                            v
                    ┌──────────────┐
                    │    MinIO     │
                    │ S3-compatible│
                    │   Storage    │
                    └──────────────┘
                Buckets: dataset, processed-data,
                        mlflow, model-promotion,
                        inference-logs
```

## Components

### Infrastructure Services

#### Kafka
- Message broker for pipeline coordination
- Single node in dev, cluster in prod
- Topics auto-created by consumers

#### MinIO
- S3-compatible object storage
- Stores datasets, models, inference logs
- Init job creates required buckets

#### PostgreSQL
- MLflow backend store
- Stores experiment metadata and run info

#### MLflow
- Experiment tracking and model registry
- Web UI on port 5000
- Artifacts stored in MinIO

### Pipeline Services

#### Preprocess
- Reads raw CSV, builds processed Parquet
- Publishes claim-check messages to Kafka
- Idempotent via config-hash

#### Training
- **train-gru**: GRU neural network
- **train-lstm**: LSTM neural network
- **nonml-prophet**: Prophet baseline
- Consume training-data, log to MLflow
- Publish model-training events

#### Eval
- Waits for all model types to complete
- Scores models, promotes best
- Writes promotion pointers to MinIO
- Publishes model-selected events

#### Inference
- Scalable prediction API (FastAPI)
- Loads promoted models from MLflow
- Exposes /predict, /ready, /metrics endpoints
- Horizontal scaling + optional HPA

#### Inference-LB
- HAProxy load balancer
- Distributes requests across inference pods
- Health checks and graceful failover

### Monitoring

#### Prometheus
- Scrapes metrics from inference pods
- Stores time-series data

#### Grafana
- Visualizes metrics from Prometheus
- Pre-configured dashboards (if provisioned)

### Load Testing

#### Locust
- **Master**: Web UI + coordinator (port 8089)
- **Workers**: Distributed load generators
- Tests inference endpoints

## Scaling

### Manual Scaling

```bash
# Scale inference pods
kubectl scale deployment inference --replicas=10

# Scale Locust workers
kubectl scale deployment locust-worker --replicas=20
```

### Autoscaling (HPA)

Enable in values:
```yaml
inference:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70
```

Deploy:
```bash
helm upgrade flts-pipeline . --set inference.autoscaling.enabled=true
```

Monitor:
```bash
kubectl get hpa
kubectl describe hpa inference
```

## Troubleshooting

### Pods not starting

```bash
# Check pod status
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>

# Check persistent volumes
kubectl get pv
kubectl get pvc

# Common issues:
# - Storage provisioner not available
# - Resource limits too high for cluster
# - Image pull failures
```

### Services not accessible

```bash
# Check services
kubectl get svc
kubectl describe svc <service-name>

# Check endpoints
kubectl get endpoints

# Test internal connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
# Inside container:
wget -O- http://mlflow:5000
wget -O- http://inference:8000/ready
```

### Training not starting

```bash
# Check Kafka is healthy
kubectl exec -it deployment/kafka -- bash
kafka-topics.sh --list --bootstrap-server localhost:9092

# Check MinIO init completed
kubectl get jobs
kubectl logs job/minio-init

# Check training pod logs
kubectl logs deployment/train-gru
```

### Inference errors

```bash
# Check if model is loaded
kubectl exec -it deployment/inference -- curl http://localhost:8000/ready

# Check MLflow connection
kubectl exec -it deployment/inference -- curl http://mlflow:5000

# Check promotion pointer exists
kubectl exec -it deployment/fastapi -- ls -la /data/model-promotion/
```

## Upgrading

```bash
# Update values or templates
helm upgrade flts-pipeline . -f values-dev.yaml

# Force pod recreation
helm upgrade flts-pipeline . --force

# Rollback if needed
helm rollback flts-pipeline
```

## Uninstalling

```bash
# Delete all resources
helm uninstall flts-pipeline

# Delete persistent volumes (WARNING: deletes data)
kubectl delete pvc --all
```

## Production Checklist

- [ ] Use external secrets management (e.g., sealed-secrets, Vault)
- [ ] Configure persistent volumes with retain policy
- [ ] Enable TLS for Ingress
- [ ] Set resource requests/limits appropriately
- [ ] Enable HPA for inference
- [ ] Configure backup for PostgreSQL and MinIO
- [ ] Set up monitoring alerts in Grafana
- [ ] Use NetworkPolicies for pod-to-pod security
- [ ] Enable Pod Security Standards
- [ ] Configure node affinity/anti-affinity for HA
- [ ] Set up log aggregation (EFK, Loki)

## Development Tips

```bash
# Quick iteration: update image and restart
kubectl set image deployment/inference inference=myregistry/inference:latest
kubectl rollout restart deployment/inference

# Copy files to/from pods
kubectl cp local-file pod-name:/path/in/container
kubectl cp pod-name:/path/in/container local-file

# Execute commands in pods
kubectl exec -it deployment/preprocess -- bash
kubectl exec -it deployment/inference -- python -c "import torch; print(torch.cuda.is_available())"

# View all resource usage
kubectl top nodes
kubectl top pods
```

## Architecture Decisions

### Why Kafka?
- Decouples services
- Claim-check pattern for large datasets
- Enables replay and reprocessing

### Why MinIO?
- S3-compatible, runs in k8s
- Cheaper than cloud S3 for on-prem
- Single storage backend for all artifacts

### Why HAProxy for LB?
- More control than k8s Service alone
- Advanced health checks
- Weighted routing (future: canary deployments)

### Why separate train pods?
- Independent scaling per model type
- Fault isolation
- Parallel training

## License

See LICENSE file in repository root.

## Support

For issues and questions:
- GitHub Issues: https://github.com/harp-wing/FLTS/issues
- Documentation: See main README.md

## Contributors

See CONTRIBUTORS.md in repository root.
