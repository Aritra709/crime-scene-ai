$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\backend"
if (-not (Test-Path ".venv")) { python -m venv .venv }
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
