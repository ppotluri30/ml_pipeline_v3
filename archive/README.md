# Archive Directory

This folder contains archived files from the ML Pipeline project cleanup performed on 2025-11-13.

**⚠️ Nothing was deleted - all files were moved here for historical reference.**

## Directory Structure

| Folder | Contents |
|--------|----------|
| `reports/` | Markdown report files (validation reports, fix reports, analysis docs) |
| `load_tests/` | Load testing results, CSV telemetry, Locust output files |
| `logs/` | Text log files from various debugging/validation sessions |
| `temp_outputs/` | JSON, YAML, HTML, and Python temp files used during development |
| `old_scripts/` | PowerShell scripts that were replaced or are no longer actively used |
| `old_k8s/` | Previous `.k8s/` manifests (replaced by `.k8s-gke/`) |
| `.k8s-archive/` | Older archived K8s configurations |
| `.k8s-clean/` | Clean K8s manifest versions |
| `.kubernetes/` | Deprecated Kubernetes configs |
| `diagnostics_tmp/` | Temporary diagnostic outputs |

## Notable Archives

### Reports (55 files)
- Pipeline validation reports (`PIPELINE_*.md`)
- K8S deployment status reports (`K8S_*.md`)
- KEDA/HPA configuration reports (`KEDA_*.md`, `HPA_*.md`)
- Performance analysis (`INFERENCE_*.md`, `LOAD_TEST_*.md`)
- Fix/implementation reports (`*_FIX_*.md`, `*_IMPLEMENTATION_*.md`)

### Load Test Results
- `autoscaling_results/` - HPA/autoscaling telemetry CSVs
- `capacity_analysis/` - Capacity test telemetry
- `locust_results/` - Locust distributed test outputs
- Various timestamp-named CSVs from monitoring sessions

### Scripts (20 archived)
- Validation scripts (`validate_*.ps1`)
- Test runners (`run_*.ps1`, `test_*.ps1`)
- Build/deploy helpers (`build_*.ps1`, `deploy_*.ps1`)

## Restoration

To restore any file:
```powershell
# Example: Restore a specific report
Copy-Item "archive\reports\PIPELINE_VALIDATION_REPORT.md" ".\"

# Example: Restore all reports
Copy-Item "archive\reports\*.md" ".\reports_restored\"
```

## Safe to Delete?

These files are safe to delete if:
1. You've confirmed the current pipeline works end-to-end
2. You don't need historical debugging/validation context
3. You've backed up anything needed for compliance/auditing

Recommended retention: **30 days minimum** after deployment verification.
