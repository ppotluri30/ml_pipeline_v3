Param(
    [int]$Count = 1000,
    [int]$TtlMs = 2000,
    [string]$Url = 'http://localhost:8023/publish_inference_claims'
)

$ErrorActionPreference = 'Stop'

$body = @{ count = $Count; ttl_ms = $TtlMs } | ConvertTo-Json -Compress
Write-Host "POST $Url with TTL $TtlMs ms, count=$Count"
$resp = Invoke-RestMethod -Uri $Url -Method Post -ContentType 'application/json' -Body $body
$resp | ConvertTo-Json -Compress | Write-Output
