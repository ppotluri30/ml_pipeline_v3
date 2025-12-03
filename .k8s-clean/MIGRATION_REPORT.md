# YAML Cleanup and Standardization Report

**Generated:** 2025-12-02  
**Status:** ✅ Complete

---

## Executive Summary

Successfully reorganized 107+ scattered Kubernetes YAML files into a canonical production-grade structure:
- **14 essential resources** consolidated into `.k8s-clean/`
- **47 deprecated files** archived to `.k8s-archive/`
- **13 dead/unsafe files** archived (including HPAs that conflicted with KEDA)
- **No service disruption** - existing cluster state preserved

---

## Cluster Validation Summary

| Component | Status | Details |
|-----------|--------|---------|
| `inference-http` Deployment | ✅ Running | 2/2 pods, KEDA-managed |
| `inference-worker` Deployment | ✅ Running | 1/1 pods, Kafka consumer |
| `inference-http-rps-scaler` ScaledObject | ✅ Ready | Prometheus triggers active |
| `keda-hpa-inference-http-rps-scaler` HPA | ✅ Active | 2 current replicas |
| Prometheus Scraping | ✅ Healthy | Both inference pods scraped at `/prometheus` |
| Locust Load Testing | ✅ Ready | 1 master, 4 workers |
| Training Pipeline | ✅ Running | GRU, LSTM, Prophet trainers |
| Eval Pipeline | ✅ Running | Model evaluation with RBAC |

---

## Directory Structure

### Before (Scattered)
```
ml_pipeline_v3/
├── .k8s/                    # 30+ files, mixed states
├── .kubernetes/             # 25+ kompose-generated files  
├── (root)                   # 15+ scattered YAMLs
├── current-scaledobject.yaml
├── keda-*.yaml              # Various KEDA experiments
└── prometheus-*.yaml        # Various Prometheus patches
```

### After (Organized)
```
ml_pipeline_v3/
├── .k8s-clean/              # ← Production source of truth
│   ├── kustomization.yaml   # Deploy: kubectl apply -k .k8s-clean/
│   ├── namespace.yaml
│   ├── inference/
│   │   ├── deployment.yaml  # inference-http (HTTP-only)
│   │   └── service.yaml
│   ├── worker/
│   │   └── deployment.yaml  # inference-worker (Kafka consumer)
│   ├── autoscaling/
│   │   ├── scaledobject.yaml           # KEDA Prometheus triggers
│   │   └── prometheus-scrape-config.yaml
│   ├── locust/
│   │   ├── configmap.yaml
│   │   ├── master.yaml
│   │   └── worker.yaml
│   ├── rbac/
│   │   └── eval-rbac.yaml
│   ├── pipeline/
│   │   ├── eval.yaml
│   │   └── training.yaml    # GRU, LSTM, Prophet
│   └── INVENTORY.md         # Full classification table
├── .k8s-archive/            # ← Deprecated files (safe to delete)
│   ├── inference-hpa.yaml
│   ├── inference-http-hpa.yaml
│   ├── inference-guardrail-hpa.yaml
│   ├── keda-http-inference.yaml
│   └── ... (47 files total)
├── .k8s/                    # ← Legacy (migrate to .k8s-clean)
├── .kubernetes/             # ← Kompose-generated (deprecated)
└── .helm/                   # ← Unchanged (infrastructure)
```

---

## Classification Summary

| Category | Count | Location | Action |
|----------|-------|----------|--------|
| **A - Required** | 14 | `.k8s-clean/` | Deploy via Kustomize |
| **B - Outdated** | 35 | `.k8s-archive/` | Safe to delete after migration |
| **C - Dead/Unsafe** | 13 | `.k8s-archive/` | MUST NOT be applied |

### Category C (Dead/Unsafe) - Archived Files

These files were immediately archived because they could break the working autoscaling:

| File | Reason |
|------|--------|
| `inference-hpa.yaml` | CPU-based HPA conflicts with KEDA |
| `inference-http-hpa.yaml` | CPU-based HPA conflicts with KEDA |
| `inference-guardrail-hpa.yaml` | Secondary HPA conflicts with KEDA |
| `inference-keda-scaledobject.yaml` | Targets wrong deployment (`inference`) |
| `keda-http-inference.yaml` | HTTPScaledObject (removed approach) |
| `keda-http-proxy-service.yaml` | ExternalName for HTTP Add-on proxy |
| `keda-inference-http-redirect.yaml` | ExternalName redirect (removed) |
| `keda-backend-service.yaml` | Backend service for HTTP Add-on |
| `current-scaledobject.yaml` | Exported state targeting wrong deployment |
| `inference-deployment.yaml` (.k8s) | Old monolithic (includes Kafka consumer) |
| `inference-deployment.yaml` (.kubernetes) | Kompose-generated, wrong labels |
| `inference-service.yaml` (.kubernetes) | Wrong selector (io.kompose.service) |
| `inference-keda-scaler.yaml` | Targets wrong deployment |

---

## Dependency Graph

```
                                  ┌─────────────────┐
                                  │   namespace     │
                                  │   (default)     │
                                  └────────┬────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
           ┌────────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐
           │      rbac/      │    │   autoscaling/  │    │    pipeline/    │
           │  eval-rbac.yaml │    │ prometheus-     │    │  training.yaml  │
           └────────┬────────┘    │ scrape-config   │    └────────┬────────┘
                    │             └────────┬────────┘             │
                    │                      │                      │
           ┌────────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐
           │   pipeline/     │    │   inference/    │    │    locust/      │
           │   eval.yaml     │    │ deployment.yaml │    │   (all files)   │
           └─────────────────┘    │  service.yaml   │    └─────────────────┘
                                  └────────┬────────┘
                                           │
                                  ┌────────▼────────┐
                                  │    worker/      │
                                  │ deployment.yaml │
                                  └────────┬────────┘
                                           │
                                  ┌────────▼────────┐
                                  │  autoscaling/   │
                                  │ scaledobject    │
                                  └─────────────────┘
```

---

## Deployment Instructions

### Deploy Clean YAMLs (Kustomize)
```bash
# Full deployment
kubectl apply -k .k8s-clean/

# Dry-run validation
kubectl apply -k .k8s-clean/ --dry-run=client
```

### Verify Deployment
```bash
# Check all resources
kubectl get deployments,services,scaledobjects,hpa

# Verify autoscaling
kubectl describe scaledobject inference-http-rps-scaler

# Verify Prometheus scraping
kubectl exec deployment/prometheus-server -c prometheus-server -- \
  wget -qO- 'http://localhost:9090/api/v1/targets?state=active' | grep inference
```

---

## Migration Diff Summary

### Key Changes

1. **Deployment Split**
   - Old: Single `inference` deployment (HTTP + Kafka consumer)
   - New: `inference-http` (HTTP only) + `inference-worker` (Kafka only)

2. **Autoscaling Method**
   - Old: Various approaches (CPU HPA, KEDA HTTP Add-on, multiple ScaledObjects)
   - New: Single KEDA ScaledObject with Prometheus triggers (RPS + P95 latency)

3. **Prometheus Integration**
   - Old: Service annotations for `/metrics` (returns JSON)
   - New: Pod discovery via `inference-pods-fast` job scraping `/prometheus`

4. **Service Discovery**
   - Old: Mixed selectors (`app=inference`, `io.kompose.service`)
   - New: Consistent `app=inference-http` selector

### Resource Count Comparison

| Resource Type | Before | After | Change |
|---------------|--------|-------|--------|
| Deployment files | 15+ | 7 | -53% |
| Service files | 12+ | 3 | -75% |
| HPA files | 4 | 0 | -100% (KEDA manages) |
| ScaledObject files | 5 | 1 | -80% |
| ConfigMap files | 8+ | 2 | -75% |

---

## Post-Migration Cleanup

Once confident in the new structure, you can safely remove:

```powershell
# Remove archived files (after validating new structure works)
Remove-Item -Recurse -Force .\.k8s-archive\

# Remove legacy directories (keep .helm)
Remove-Item -Recurse -Force .\.kubernetes\

# Clean root-level scattered YAMLs (check each first)
Get-ChildItem -Path . -Name "*.yaml" | Where-Object { 
    $_ -notmatch "docker-compose|values" 
} | Remove-Item -WhatIf
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `.k8s-clean/INVENTORY.md` | Full classification table with all 107 files |
| `.k8s-clean/kustomization.yaml` | Kustomize deployment manifest |
| `scripts/Run-FullLoadTest.ps1` | Parallel telemetry load test script |
| `KEDA_HTTP_AUTOSCALING_REPORT.md` | Detailed KEDA configuration documentation |

---

## Validation Checklist

- [x] All Category C files archived
- [x] Dry-run validation passes (`kubectl apply -k .k8s-clean/ --dry-run=client`)
- [x] Prometheus scraping both inference-http pods (health: UP)
- [x] KEDA ScaledObject status: Ready
- [x] HPA controlled by KEDA (not manual CPU HPA)
- [x] inference-http deployment: 2/2 replicas running
- [x] inference-worker deployment: 1/1 replica running
- [x] Training pipeline: GRU, LSTM, Prophet running
- [x] Eval pipeline: Running with RBAC
- [x] Locust: 1 master, 4 workers ready

**Migration Status: ✅ COMPLETE**
