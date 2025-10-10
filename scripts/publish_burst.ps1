Param(
    [int]$Count = 10000,
    [string]$Url = 'http://localhost:8023/publish_inference_claims'
)

$ErrorActionPreference = 'Stop'

$body = @{ count = $Count } | ConvertTo-Json -Compress
Write-Host "POST $Url burst count=$Count"
$resp = Invoke-RestMethod -Uri $Url -Method Post -ContentType 'application/json' -Body $body
$resp | ConvertTo-Json -Compress | Write-Output
