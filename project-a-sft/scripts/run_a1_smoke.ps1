$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
python (Join-Path $PSScriptRoot "generate_a1_smoke.py") --output (Join-Path $projectRoot "artifacts\a1_smoke")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
