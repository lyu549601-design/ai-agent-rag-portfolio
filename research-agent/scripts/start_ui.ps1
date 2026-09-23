$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Test-Path ".env")) {
    Write-Host "请先运行 .\scripts\setup.ps1 并配置 .env"
    exit 1
}
docker compose up -d --wait
$listener = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if (-not $listener) {
    Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","research_agent.web_app:app","--host","127.0.0.1","--port","8501" -WorkingDirectory $root -WindowStyle Hidden
    Start-Sleep -Seconds 10
}
Start-Process "http://127.0.0.1:8501"
Write-Host "控制台已启动：http://127.0.0.1:8501"