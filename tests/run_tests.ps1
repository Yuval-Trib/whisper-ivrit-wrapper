# Run all handler modes discovered from test_<mode>.json files.
# Usage (from repo root): .\tests\run_tests.ps1
# Adding a new endpoint: add it to prepare.py INPUTS dict, runner picks it up automatically.

$root = Split-Path $PSScriptRoot -Parent
$image = "whisper-ivrit-wrapper"

Write-Host "Generating test inputs..."
python "$PSScriptRoot\prepare.py"
if ($LASTEXITCODE -ne 0) { exit 1 }

$results = @{}

Get-ChildItem "$PSScriptRoot\test_*.json" | ForEach-Object {
    $mode = $_.BaseName -replace "^test_", ""
    Write-Host ""
    Write-Host "Testing $mode..."

    $output = docker run --rm --gpus all `
        --env-file "$root\.env" `
        -e "MODE=$mode" `
        -e "DEVICE=cuda" `
        -e "HF_HOME=/tmp/hf" `
        -v "$($_.FullName):/app/test_input.json" `
        $image 2>&1

    $output | Write-Host

    $passed = $output | Select-String "Job local_test completed successfully"
    $results[$mode] = [bool]$passed
}

Write-Host ""
Write-Host "Results:"
foreach ($mode in $results.Keys | Sort-Object) {
    $status = if ($results[$mode]) { "PASS" } else { "FAIL" }
    Write-Host "  $mode : $status"
}

$anyFailed = $results.Values | Where-Object { -not $_ }
if ($anyFailed) { exit 1 } else { exit 0 }
