# Scripts Directory

Essential PowerShell scripts for operating and testing the ML Pipeline.

## Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `k8s_auto_hpa_tests.ps1` | Automated HPA matrix testing with CSV/Markdown reporting | `.\k8s_auto_hpa_tests.ps1 -TestDuration 120 -UserCounts @(50,100,200)` |
| `Monitor-LiveLatency.ps1` | Real-time latency monitoring during load tests | `.\Monitor-LiveLatency.ps1` (requires Locust UI port-forward) |
| `monitor_keda_scaling.ps1` | KEDA scaling event tracker with Prometheus queries | `.\monitor_keda_scaling.ps1` |
| `run_all_locust_tests.ps1` | Docker Compose load test matrix automation | `.\run_all_locust_tests.ps1 -TestDuration 60 -ReplicaCounts @(1,2,4)` |

## Prerequisites

- PowerShell 5.1+ (Windows) or PowerShell Core (cross-platform)
- `kubectl` configured and connected to cluster
- Docker Desktop (for `run_all_locust_tests.ps1`)

## Quick Start

### Run HPA Validation Tests (Kubernetes)
```powershell
# Basic test
.\scripts\k8s_auto_hpa_tests.ps1 -TestDuration 60 -UserCounts @(100)

# Full matrix
.\scripts\k8s_auto_hpa_tests.ps1 -TestDuration 120 -UserCounts @(50,100,200) -WorkerCounts @(4,8) -HPATargetCPU 60
```

### Monitor KEDA Scaling in Real-Time
```powershell
# Start monitoring (run in separate terminal)
.\scripts\monitor_keda_scaling.ps1
```

### Run Docker Compose Load Tests
```powershell
# Matrix test across replica/user combinations
.\scripts\run_all_locust_tests.ps1 -TestDuration 60 -ReplicaCounts @(1,2,4) -WorkerCounts @(4) -UserCounts @(50,100)
```

## Output Locations

- HPA tests: `reports/k8s_hpa_performance/*.csv`, `*.md`
- KEDA monitoring: CSV output with timestamps
- Locust tests: `locust/results/auto_matrix/auto_summary.csv`

## Notes

- Scripts in `archive/old_scripts/` were moved during cleanup - check there if something is missing
- All scripts support `-Help` or `Get-Help` for documentation
- Modify `$env:KUBECONFIG` before running K8s scripts if using non-default context
