"""项目路径与配置。

项目根目录按 ``__file__`` 反推，因此无论从哪个工作目录启动都能找到数据与输出目录。
密钥只从环境变量（或 .env 文件）读取，不写进代码。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

ENV_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    evals: Path
    reports: Path
    workspace: Path

    @property
    def audit_log(self) -> Path:
        return self.workspace / "audit" / "audit_log.jsonl"

    @property
    def outbox(self) -> Path:
        return self.workspace / "outbox"

    def ensure(self) -> "Paths":
        for directory in (self.workspace, self.audit_log.parent, self.outbox, self.reports):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def default_paths(root: Path | None = None) -> Paths:
    base = (root or project_root()).resolve()
    return Paths(
        root=base,
        data=base / "data",
        evals=base / "evals",
        reports=base / "reports",
        workspace=base / "workspace",
    )


def load_env_file(path: str | Path | None = None) -> list[str]:
    """读取 .env 文件并写入环境变量（不覆盖已存在的变量）。返回载入的键名。"""
    if path is None:
        path = project_root() / ".env"
    path = Path(path)
    if not path.exists():
        return []

    loaded: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = ENV_KEY_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded