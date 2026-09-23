param([switch]$Local)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "已创建 .env，请填入 DEEPSEEK_API_KEY 后重跑。"
    exit 1
}

$localArg = @()
if (-not $Local) {
    docker compose up -d --wait
} else {
    $localArg = @("--local")
}

.\.venv\Scripts\python.exe -m research_agent.cli ingest @localArg
.\.venv\Scripts\python.exe -m research_agent.cli demo @localArg --topic "企业级 AI Agent 技术现状与趋势"
.\.venv\Scripts\python.exe -m research_agent.cli eval-retrieval @localArg