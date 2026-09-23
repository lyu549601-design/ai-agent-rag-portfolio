#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "请先填写 .env 中的 DEEPSEEK_API_KEY"
  exit 1
fi
local_args=()
if [ "${LOCAL:-0}" = "1" ]; then
  local_args+=(--local)
else
  docker compose up -d --wait
fi
./.venv/bin/python -m research_agent.cli ingest "${local_args[@]}"
./.venv/bin/python -m research_agent.cli demo "${local_args[@]}" --topic "企业级 AI Agent 技术现状与趋势"
./.venv/bin/python -m research_agent.cli eval-retrieval "${local_args[@]}"