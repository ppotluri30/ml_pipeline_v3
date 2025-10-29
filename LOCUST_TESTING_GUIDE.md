# Automated Locust Testing Quick Start Guide

## Prerequisites

✅ **Verify these services are running:**

```powershell
docker compose ps
```

Required containers:
- `kafka` (Healthy)
- `minio` (Running)
- `mlflow` (Running)
- `fastapi-app` (Running)
- `inference` (1+ replicas, Healthy)
- `inference-lb` (Healthy)
- `locust` (Running - Master)
- `locust-worker` (1+ workers, Running)

## Quick Start

### 1. Ensure Locust Master + Workers are Running

```powershell
# Start Locust master and 4 workers
docker compose up -d --scale locust-worker=4 locust locust-worker

# Wait a few seconds and verify
docker compose logs locust --tail 20
```

You should see: `"4 workers connected"`

### 2. Run the Automated Test Suite

```powershell
# Run with default settings (2 min tests, all configs)
.\run_all_locust_tests.ps1

# Or customize:
.\run_all_locust_tests.ps1 -TestDuration 180 -UserCounts @(100,200,400)
```

### 3. Monitor Progress

The script will display:
- ✅ Live progress bars for each test
- 🏃 Real-time RPS and latency updates every second
- 📊 Summary after each test completes
- 🎉 Final comprehensive results table

### 4. Review Results

After completion, check:
- **CSV**: `locust/results/auto_matrix/auto_summary.csv`
- **Markdown Report**: `locust/results/auto_matrix/auto_summary.md`
- **Individual Tests**: `locust/results/auto_matrix/replicas{N}_workers{M}_u{users}/`

## Script Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TestDuration` | 120 | Seconds per test |
| `ReplicaCounts` | @(1,2,4,8) | Inference replicas to test |
| `WorkerCounts` | @(4,8) | Locust worker counts |
| `UserCounts` | @(200,400,800) | Concurrent users per test |
| `ResultsBaseDir` | `locust/results/auto_matrix` | Output directory |
| `LocustMasterUrl` | `http://localhost:8089` | Locust API endpoint |

## Example Custom Run

```powershell
# Quick test with shorter duration
.\run_all_locust_tests.ps1 -TestDuration 60 -UserCounts @(100,200)

# Focus on specific replica counts
.\run_all_locust_tests.ps1 -ReplicaCounts @(4,8) -WorkerCounts @(8)

# Full matrix with extended duration
.\run_all_locust_tests.ps1 -TestDuration 300
```

## What the Script Does

### For Each Configuration:

1. **Scale Infrastructure**
   - Scales inference containers to target replica count
   - Scales Locust workers to target count
   - Waits for health checks

2. **Run Test**
   - Resets Locust statistics
   - Starts distributed test with configured users
   - Monitors live RPS/latency every second
   - Displays color-coded progress bar

3. **Collect Results**
   - Extracts metrics from Locust API
   - Saves to CSV and individual directories
   - Displays formatted summary

4. **Generate Reports**
   - Creates comprehensive Markdown report
   - Includes performance analysis
   - Identifies optimal configurations

## Live Output Example

```
============================================================
🧪 TEST: replicas4_workers4_u400
   Inference Replicas: 4
   Locust Workers: 4
   Concurrent Users: 400
   Duration: 120s
============================================================
🏃 Starting test: 400 users at 26/s spawn rate...
🏃 Ramping up users (16s)...
🏃 Test running for 120s with live monitoring...

[Replicas=4 Workers=4 Users=400] [████████████████░░░░] 80% | RPS: 45.32 | Med: 156ms | P95: 420ms | Reqs: 5438 | Fails: 0

📊 Test Results:
   Total Requests:  5438
   RPS:             45.32
   Median:          156ms
   P95:             420ms
   P99:             580ms
   Failures:        0 (0%)
   
✅ Test completed: replicas4_workers4_u400
```

## Troubleshooting

### Locust Master Not Ready
```powershell
# Restart Locust services
docker compose restart locust
docker compose up -d --scale locust-worker=4 locust-worker
```

### Inference Containers Not Healthy
```powershell
# Check inference logs
docker compose logs inference --tail 50

# Restart inference if needed
docker compose restart inference inference-lb
```

### Script Interrupted
The script can be safely rerun - it will start from the beginning. Results from previous runs are preserved in separate timestamp directories.

## Post-Test Analysis

### View Results Table
```powershell
Get-Content locust/results/auto_matrix/auto_summary.md
```

### Parse CSV for Specific Metrics
```powershell
Import-Csv locust/results/auto_matrix/auto_summary.csv | 
    Where-Object { $_.Replicas -eq 4 } | 
    Format-Table
```

### Find Best Configuration
```powershell
Import-Csv locust/results/auto_matrix/auto_summary.csv | 
    Sort-Object RPS -Descending | 
    Select-Object -First 5 | 
    Format-Table
```

## Next Steps

1. **Review the Markdown report** for performance insights
2. **Identify optimal replica count** for your workload
3. **Compare 4 vs 8 worker configurations** to understand scaling
4. **Use CSV data** for custom analysis or visualization

---

🎯 **Goal**: Determine the optimal number of inference replicas and Locust workers for maximum throughput with acceptable latency.

💡 **Tip**: Run the script during off-peak hours for most accurate results.
