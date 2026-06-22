# Test stt_long against the live RunPod endpoint.
# Usage: .\tests\stt_live_test.ps1 -AudioPath "C:\path\to\file.mp3"
param(
    [Parameter(Mandatory)][string]$AudioPath,
    [string]$Language = "he"
)

$configPath = Join-Path $PSScriptRoot "endpoints.local"
if (-not (Test-Path $configPath)) {
    Write-Error "Missing $configPath - copy endpoints.local.example, fill it in, save as endpoints.local"
    exit 1
}

$cfg = @{}
Get-Content $configPath | Where-Object { $_ -match "^\s*\w+=.+" } | ForEach-Object {
    $parts = $_ -split "=", 2
    $cfg[$parts[0].Trim()] = $parts[1].Trim()
}

$apiKey    = $cfg["RUNPOD_API_KEY"]
$endpoint  = $cfg["STT_LONG_ENDPOINT_ID"]

if (-not $apiKey -or -not $endpoint) {
    Write-Error "RUNPOD_API_KEY and STT_LONG_ENDPOINT_ID must be set in endpoints.local"
    exit 1
}


if (-not (Test-Path $AudioPath)) {
    Write-Error "Audio file not found: $AudioPath"
    exit 1
}

Write-Host "Encoding audio ($AudioPath)..."
$bytes = [System.IO.File]::ReadAllBytes($AudioPath)
$b64   = [Convert]::ToBase64String($bytes)
Write-Host "Encoded: $([math]::Round($bytes.Length / 1MB, 1)) MB"

$body = @{ input = @{ audio = $b64; language = $Language } } | ConvertTo-Json -Depth 3

Write-Host "Submitting to endpoint $endpoint..."
$t0 = Get-Date

$submit = Invoke-RestMethod `
    -Uri "https://api.runpod.ai/v2/$endpoint/run" `
    -Method Post `
    -Headers @{ Authorization = "Bearer $apiKey" } `
    -ContentType "application/json" `
    -Body $body

$jobId = $submit.id
Write-Host "Job ID: $jobId"
Write-Host ""

do {
    Start-Sleep -Seconds 5
    $poll = Invoke-RestMethod `
        -Uri "https://api.runpod.ai/v2/$endpoint/status/$jobId" `
        -Headers @{ Authorization = "Bearer $apiKey" }
    $elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    Write-Host "[$elapsed s] $($poll.status)"
} while ($poll.status -notin @("COMPLETED", "FAILED", "CANCELLED"))

$total = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
Write-Host ""

if ($poll.status -ne "COMPLETED") {
    Write-Error "Job $($poll.status): $($poll.error)"
    exit 1
}

Write-Host "Completed in $total seconds"

$outPath = Join-Path $PSScriptRoot "responses\stt_long_result.json"
$poll | ConvertTo-Json -Depth 10 | Out-File $outPath -Encoding utf8
Write-Host "Saved: $outPath"

$segments = $poll.output.segments
if ($segments) {
    Write-Host ""
    Write-Host "First 3 segments:"
    $segments | Select-Object -First 3 | ForEach-Object {
        Write-Host "  [$($_.start)s - $($_.end)s] $($_.speaker): $($_.text)"
    }
    Write-Host "  ... ($($segments.Count) total segments)"
}
