# 🚀 Automated Locust Testing - Ready to Run!

## ✅ Setup Complete!

Your distributed Locust testing environment is fully configured and ready to execute.

### Current Status

```
✅ Locust Master:     Running (flts-main-locust-1)
✅ Locust Workers:    4 workers connected and ready
✅ Inference LB:      Healthy at http://localhost:8023
✅ Inference Replicas: 1 currently running (can scale to 8)
✅ Test Script:       run_all_locust_tests.ps1 created
✅ Results Directory: locust/results/auto_matrix/ ready
```

---

## 🎯 Execute the Full Test Suite

### Run Now!

```powershell
.\run_all_locust_tests.ps1
```

This will automatically:
1. Test **4 inference replica counts**: 1, 2, 4, 8
2. Test **2 worker configurations**: 4 workers, then 8 workers
3. Test **3 user loads**: 200, 400, 800 concurrent users
4. Run **24 total tests** (4 × 2 × 3)
5. Duration: **2 minutes per test** = ~50 minutes total

### Quick Test (Optional)

If you want a faster test run:

```powershell
# Only test 4 and 8 replicas with 200 users (60s each)
.\run_all_locust_tests.ps1 `
    -TestDuration 60 `
    -ReplicaCounts @(4,8) `
    -WorkerCounts @(4) `
    -UserCounts @(200)
```

---

## 📊 What You'll Get

### Real-Time Monitoring

```
[Replicas=4 Workers=4 Users=400] [████████████████░░░░] 80% | RPS: 45.32 | Med: 156ms | P95: 420ms | Reqs: 5438
```

### After Each Test

```
📊 Test Results:
   Total Requests:  5438
   RPS:             45.32
   Median:          156ms
   P95:             420ms
   P99:             580ms
   Failures:        0 (0%)
```

### Final Summary Table

```
+------------+---------+-------+---------+-------------+-----------+-----------+-------------+
| Replicas   | Workers | Users | RPS     | Median (ms) | P95 (ms)  | P99 (ms)  | Fail (%)    |
+------------+---------+-------+---------+-------------+-----------+-----------+-------------+
| 1          | 4       | 200   | 22.45   | 180         | 450       | 620       | 0           |
| 1          | 4       | 400   | 24.18   | 320         | 780       | 980       | 0           |
| ... (all 24 configurations)
+------------+---------+-------+---------+-------------+-----------+-----------+-------------+
```

### Generated Files

1. **CSV Data**: `locust/results/auto_matrix/auto_summary.csv`
   - Machine-readable format for analysis
   - Import into Excel, Python pandas, etc.

2. **Markdown Report**: `locust/results/auto_matrix/auto_summary.md`
   - Comprehensive analysis with:
     - Performance scaling charts
     - Latency comparisons
     - Key findings and recommendations
     - Optimal configuration identification

3. **Individual Test Directories**: `locust/results/auto_matrix/replicas{N}_workers{M}_u{users}/`
   - Detailed logs for each configuration

---

## 🔍 Key Questions the Tests Will Answer

1. **How does throughput scale with inference replicas?**
   - Does 2x replicas = 2x RPS?
   - Where does scaling plateau?

2. **What's the optimal replica count for our workload?**
   - Best RPS per dollar
   - Acceptable latency thresholds

3. **Do we need 8 Locust workers or is 4 enough?**
   - Compare 4-worker vs 8-worker RPS
   - Identify client-side bottlenecks

4. **What happens under high load?**
   - Latency degradation patterns
   - Failure rates at 800 concurrent users
   - System stability metrics

5. **What's our capacity limit?**
   - Maximum sustainable RPS
   - Memory/CPU utilization
   - Network bandwidth limits

---

## 📈 Expected Performance Patterns

Based on your earlier test results:

### Current Baseline (40 users, 1 replica)
- RPS: ~15 req/s
- Median: 110ms
- P95: 350ms

### Expected with Full Tests
- **1 Replica**: 20-30 RPS @ 200 users
- **2 Replicas**: 35-50 RPS @ 400 users
- **4 Replicas**: 60-90 RPS @ 800 users
- **8 Replicas**: 100-150 RPS @ 800 users (may plateau)

*Actual results will depend on CPU, model complexity, and network*

---

## 🎬 Ready to Start?

### Pre-Flight Check

```powershell
# Verify everything is healthy
docker compose ps

# Check Locust web UI (optional)
Start-Process "http://localhost:8089"

# Ensure you have disk space for results
Get-PSDrive C
```

### Launch Tests

```powershell
# Full test suite (~50 minutes)
.\run_all_locust_tests.ps1

# The script will:
# - Display colored progress bars ✅
# - Show live RPS/latency updates 📊  
# - Auto-scale containers 🔧
# - Generate comprehensive reports 📝
# - Handle errors gracefully 🛡️
```

### While Tests Run

- ☕ Grab coffee - tests take ~50 minutes
- 👀 Watch the live updates in the console
- 📊 Monitor Locust web UI: http://localhost:8089
- 📈 Check inference metrics: http://localhost:8023/metrics

---

## 🔧 Troubleshooting

### If Script Fails to Start

```powershell
# Restart Locust services
docker compose restart locust
docker compose up -d --scale locust-worker=4 locust-worker
Start-Sleep -Seconds 10

# Verify workers connected
docker compose logs locust --tail 10

# Re-run script
.\run_all_locust_tests.ps1
```

### If Inference Unhealthy

```powershell
# Check logs
docker compose logs inference --tail 50

# Restart if needed
docker compose restart inference inference-lb
Start-Sleep -Seconds 15
```

### Script Interrupted

Just re-run it! The script creates new result directories for each run, so previous data won't be overwritten.

---

## 📚 Documentation

- **Quick Start Guide**: `LOCUST_TESTING_GUIDE.md`
- **Script Source**: `run_all_locust_tests.ps1`
- **Results Summary**: `locust/results/auto_matrix/auto_summary.md` (after tests)

---

## 🎯 Success Criteria

At the end of the test run, you should have:

✅ **24 successful test completions** (or 6 for quick test)  
✅ **Zero or near-zero failure rates** across all configurations  
✅ **Clear performance trends** showing scaling behavior  
✅ **Optimal replica count identified** for your workload  
✅ **Comprehensive CSV and Markdown reports** for analysis  

---

## 🚀 Let's Go!

```powershell
# Run the full automated test suite
.\run_all_locust_tests.ps1
```

**Estimated completion time**: ~50 minutes  
**Output**: Comprehensive performance analysis with 24 configurations  
**Next steps**: Review `auto_summary.md` for insights and recommendations

---

*Generated for the FLTS ML Pipeline distributed load testing*  
*Last Updated: 2025-10-29*
