param([switch]$Local, [int]$Limit = 12)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$localArg = @()
if (-not $Local) {
    docker compose up -d --wait
} else {
    $localArg = @("--local")
}

.\.venv\Scripts\python.exe -m research_agent.cli eval-retrieval @localArg
.\.venv\Scripts\python.exe -m research_agent.cli eval-memory @localArg --limit $Limit