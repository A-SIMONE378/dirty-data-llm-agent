$ErrorActionPreference = "Stop"
python -m pytest -v
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
