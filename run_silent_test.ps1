param([int]$Users = 150, [int]$SpawnRate = 10, [int]$Duration = 120, [int]$SampleInterval = 5)
$ErrorActionPreference = 'Continue'; $ProgressPreference = 'SilentlyContinue'
function Get-PromP95 { try { $q = [uri]::EscapeDataString('histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[2m])) by (le))'); $r = kubectl exec prometheus-server-55dbc8856c-c8n52 -c prometheus-server -- wget -qO- `"http://localhost:9090/api/v1/query?query=$q`" 2>$null | ConvertFrom-Json; $v = $r.data.result[0].value[1]; if ($v -and $v -ne 'NaN') { return [math]::Round([double]$v * 1000, 0) }; return 'N/A' } catch { return 'N/A' } }
function Get-KEDA { try { $keda = kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/default/s0-prometheus?labelSelector=scaledobject.keda.sh%2Fname%3Dinference-slo-scaler" 2>$null | ConvertFrom-Json; $val = $keda.items[0].value; if ($val -match '(-?\d+)m') { return $matches[1] }; return 'N/A' } catch { return 'N/A' } }
function Get-HPA { try { $h = kubectl get hpa keda-hpa-inference-slo-scaler -o json 2>$null | ConvertFrom-Json; return @{Current = $h.status.currentReplicas; Desired = $h.status.desiredReplicas} } catch { return @{Current = 'N/A'; Desired = 'N/A'} } }
function Get-Pods { try { $t = kubectl top pods -l app=inference --no-headers 2>$null; if (-not $t) { return @{CPU = 'N/A'; Memory = 'N/A'} }; $c = 0; $m = 0; $n = 0; foreach ($l in $t) { if ($l -match '(\d+)m\s+(\d+)Mi') { $c += [int]$matches[1]; $m += [int]$matches[2]; $n++ } }; if ($n -gt 0) { return @{CPU = [math]::Round($c / $n, 0); Memory = [math]::Round($m / $n, 0)} }; return @{CPU = 'N/A'; Memory = 'N/A'} } catch { return @{CPU = 'N/A'; Memory = 'N/A'} } }
function Get-Queue { try { $q = [uri]::EscapeDataString('avg(inference_queue_len)'); $r = kubectl exec prometheus-server-55dbc8856c-c8n52 -c prometheus-server -- wget -qO- "http://localhost:9090/api/v1/query?query=$q" 2>$null | ConvertFrom-Json; $v = $r.data.result[0].value[1]; if ($v) { return [math]::Round([double]$v, 1) }; return 0 } catch { return 0 } }
$log = @(); $events = @(); $last = 0
Write-Host "
========================================" -Fore Cyan; Write-Host "SILENT AUTOSCALING TEST - $Users users, s" -Fore Cyan; Write-Host "========================================
" -Fore Cyan
Write-Host '[00:00] Resetting to 3 replicas...' -Fore Gray; kubectl scale deployment inference --replicas=3 2>&1 | Out-Null; Start-Sleep 5
Write-Host '[00:05] Finding Locust master pod...' -Fore Gray; $locustPod = (kubectl get pods -l component=master -o jsonpath='{.items[0].metadata.name}' 2>$null); if (-not $locustPod) { $locustPod = (kubectl get pods -o name | Select-String 'locust-master' | Select-Object -First 1).ToString().Replace('pod/','') }; if (-not $locustPod) { Write-Host 'ERROR: Locust master pod not found' -Fore Red; exit 1 }; Write-Host "[00:05] Using pod: $locustPod" -Fore Gray
Write-Host '[00:05] Starting Locust (SILENT)...' -Fore Gray
$job = Start-Job { param($pod,$u,$s,$d) kubectl exec $pod -- sh -c `"env LOCUST_LOGLEVEL=CRITICAL locust --headless --host=http://inference:8000 -u $u -r $s -t ${d}s --only-summary`" 2>&1 | Out-Null } -Args $locustPod,$Users,$SpawnRate,$Duration
Write-Host "[00:05] Locust Job ID: $($job.Id) - Monitoring for s
" -Fore Green
Write-Host ('{0,-8} {1,-8} {2,-8} {3,-10} {4,-10} {5,-15} {6,-15} {7,-8}' -f 'Time','Replica','Desired','CPU(m)','Mem(Mi)','PromP95(ms)','KEDALat(ms)','Queue') -Fore Cyan
Write-Host ('{0,-8} {1,-8} {2,-8} {3,-10} {4,-10} {5,-15} {6,-15} {7,-8}' -f '--------','--------','--------','----------','----------','---------------','---------------','--------') -Fore Cyan
$start = Get-Date; $samples = [math]::Ceiling($Duration / $SampleInterval)
for ($i = 0; $i -lt $samples; $i++) {
    $e = [math]::Round(((Get-Date) - $start).TotalSeconds, 0); $m = [int][math]::Floor($e/60); $s = [int]($e%60); $ts = '{0:D2}:{1:D2}' -f $m, $s
    $p95 = Get-PromP95; $keda = Get-KEDA; $hpa = Get-HPA; $pods = Get-Pods; $q = Get-Queue
    $log += [PSCustomObject]@{Time=$ts; Replicas=$hpa.Current; Desired=$hpa.Desired; CPU=$pods.CPU; Mem=$pods.Memory; P95=$p95; KEDA=$keda; Queue=$q}
    if ($hpa.Current -ne $last -and $last -ne 0) { $events += [PSCustomObject]@{Time=$ts; From=$last; To=$hpa.Current} }; $last = $hpa.Current
    $color = if ($hpa.Current -ne $hpa.Desired) { 'Yellow' } else { 'White' }
    Write-Host ('{0,-8} {1,-8} {2,-8} {3,-10} {4,-10} {5,-15} {6,-15} {7,-8}' -f $ts,$hpa.Current,$hpa.Desired,$pods.CPU,$pods.Memory,$p95,$keda,$q) -Fore $color
    if ($e -ge $Duration) { break }; Start-Sleep $SampleInterval
}
Write-Host "
[$ts] Waiting for Locust..." -Fore Gray; $job | Wait-Job -Timeout 30 | Out-Null
if ($job.State -eq 'Completed') { Write-Host "[$ts] Locust completed" -Fore Green; $out = Receive-Job $job 2>&1; if ($out) { Write-Host "
Locust:" -Fore Yellow; Write-Host $out } } else { Write-Host "[$ts] Locust timeout (State: $($job.State))" -Fore Yellow }
Remove-Job $job -Force 2>$null
Write-Host "
========================================" -Fore Cyan; Write-Host 'SUMMARY' -Fore Cyan; Write-Host "========================================" -Fore Cyan
$valid = $log | Where-Object { $_.P95 -ne 'N/A' } | Select-Object -ExpandProperty P95
if ($valid) { $avg = [math]::Round(($valid|Measure-Object -Average).Average,0); $max = ($valid|Measure-Object -Maximum).Maximum; $min = ($valid|Measure-Object -Minimum).Minimum; Write-Host "
Latency P95: Avg=${avg}ms | Min=${min}ms | Max=${max}ms" -Fore Yellow }
if ($events.Count -gt 0) { Write-Host "
Scaling Events:" -Fore Yellow; foreach ($e in $events) { $dir = if ($e.To -gt $e.From) { 'UP' } else { 'DOWN' }; Write-Host "  [$($e.Time)] $dir $($e.From) -> $($e.To) replicas" -Fore Green } } else { Write-Host "
Scaling: Stable at $last replicas" -Fore Yellow }
$maxR = ($log | Where-Object { $_.Replicas -ne 'N/A' } | Select-Object -ExpandProperty Replicas | Measure-Object -Maximum).Maximum
Write-Host "
Peak Replicas: $maxR" -Fore Yellow
$csv = "autoscaling_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"; $log | Export-Csv $csv -NoType
Write-Host "
Metrics exported: $csv" -Fore Cyan; Write-Host "========================================
" -Fore Cyan
