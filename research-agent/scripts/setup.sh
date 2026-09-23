#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
if [ ! -f .env ]; then
  cp .env.example .env
  echo "已创建 .env，请填入 DEEPSEEK_API_KEY。"
fi