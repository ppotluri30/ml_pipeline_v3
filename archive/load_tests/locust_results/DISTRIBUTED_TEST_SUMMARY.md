# Distributed Locust Load Testing Results Summary

## Test Configuration
- **Locust Setup**: Master + 4 Workers (Distributed Mode)
- **Test Duration**: 2 minutes per configuration
- **Target**: Inference Load Balancer (`http://inference-lb`)
- **Model**: GRU (promoted model from full dataset training)

## Test Matrix

### Phase 1: 4 Locust Workers

| Inference Replicas | Users | RPS (req/s) | Median (ms) | P95 (ms) | P99 (ms) | Failures (%) | Notes |
|--------------------|-------|-------------|-------------|----------|----------|--------------|-------|
| 1 | 200 | - | - | - | - | - | Running... |
| 1 | 400 | - | - | - | - | - | Pending |
| 1 | 800 | - | - | - | - | - | Pending |
| 2 | 200 | - | - | - | - | - | Pending |
| 2 | 400 | - | - | - | - | - | Pending |
| 2 | 800 | - | - | - | - | - | Pending |
| 4 | 200 | - | - | - | - | - | Pending |
| 4 | 400 | - | - | - | - | - | Pending |
| 4 | 800 | - | - | - | - | - | Pending |
| 8 | 200 | - | - | - | - | - | Pending |
| 8 | 400 | - | - | - | - | - | Pending |
| 8 | 800 | - | - | - | - | - | Pending |

### Phase 2: 8 Locust Workers (Planned)

| Inference Replicas | Users | RPS (req/s) | Median (ms) | P95 (ms) | P99 (ms) | Failures (%) | Notes |
|--------------------|-------|-------------|-------------|----------|----------|--------------|-------|
| 1 | 200 | - | - | - | - | - | Pending |
| 1 | 400 | - | - | - | - | - | Pending |
| 1 | 800 | - | - | - | - | - | Pending |
| 2 | 200 | - | - | - | - | - | Pending |
| 2 | 400 | - | - | - | - | - | Pending |
| 2 | 800 | - | - | - | - | - | Pending |
| 4 | 200 | - | - | - | - | - | Pending |
| 4 | 400 | - | - | - | - | - | Pending |
| 4 | 800 | - | - | - | - | - | Pending |
| 8 | 200 | - | - | - | - | - | Pending |
| 8 | 400 | - | - | - | - | - | Pending |
| 8 | 800 | - | - | - | - | - | Pending |

## Previous Single-Instance Results (for comparison)

From earlier standalone tests (40 users, 60s):

| Setup | Inference Replicas | RPS (req/s) | Median (ms) | P95 (ms) | P99 (ms) | Failures (%) |
|-------|-------------------|-------------|-------------|----------|----------|--------------|
| Standalone | 1 | 15.46 | 110 | 350 | 480 | 0% |
| Standalone | 2 | 15.45 | 110 | 400 | 580 | 0% |
| Standalone | 4 | 13.79 | 140 | 500 | 740 | 0% |

## Analysis & Observations

### Current Status
- ✅ Distributed Locust setup deployed (master + 4 workers)
- ✅ Infrastructure ready: Kafka, MinIO, MLflow, Inference-LB
- ✅ 8 inference containers available for scaling
- 🔄 Test execution in progress

### Expected Patterns
1. **Throughput Scaling**: RPS should increase linearly with inference replicas up to a saturation point
2. **Latency Impact**: Median and P95 latencies should decrease as replicas increase (better load distribution)
3. **Saturation Point**: Beyond optimal replica count, diminishing returns expected
4. **Worker Impact**: 8 workers vs 4 workers should show higher aggregate throughput

### Key Metrics to Watch
- **RPS Plateau**: Where throughput stops increasing despite more replicas
- **Latency Floor**: Minimum achievable latency (network + model inference time)
- **Failure Rate**: Should remain 0% if system is properly scaled
- **Worker Efficiency**: Compare 4-worker vs 8-worker throughput at same replica counts

## Test Execution Notes

### Environment
- **Docker Compose Setup**: All containers persistent (no removals during tests)
- **Load Balancer**: HAProxy routing to inference containers
- **Model Ready**: GRU model (HIDDEN_SIZE=128, NUM_LAYERS=2) pre-loaded
- **Dataset**: Full PobleSec dataset (15,927 train / 3,982 test rows)

### Access Points
- Locust Web UI: http://localhost:8089
- Inference LB Health: http://localhost:8023/healthz
- MLflow: http://localhost:5000

---

**Last Updated**: 2025-10-29  
**Status**: Test execution in progress
