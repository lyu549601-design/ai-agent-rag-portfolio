"""执行沙箱。

写操作在通过策略校验与人工审批后，才由沙箱落地执行：

* ``LocalSandbox``（默认）：把执行结果写入工作区 ``outbox`` 目录，模拟"写入真实系统"，
  不接触任何外部系统；
* ``DockerSandbox``：用一次性容器执行，容器无网络、根文件系统只读、只挂载 outbox 目录，
  用于演示"最小权限执行"。

沙箱自身不做权限判断，也不做降级兜底：容器不可用时按 fail-closed 处理，直接返回失败。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


@dataclass
class SandboxResult:
    ok: bool
    backend: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    artifact: str | None = None
    stdout: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Sandbox(Protocol):
    name: str

    def execute(self, operation: dict[str, Any]) -> SandboxResult: ...

    def probe(self) -> tuple[bool, str]: ...


def _stamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class LocalSandbox:
    """默认后端：把结果写成工作区里的 JSON 文件，表示"已提交到目标系统"。"""

    name = "local"

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.outbox = self.workspace / "outbox"
        self.outbox.mkdir(parents=True, exist_ok=True)

    def probe(self) -> tuple[bool, str]:
        return True, "本地沙箱可用（写入 workspace/outbox，模拟目标系统）"

    def execute(self, operation: dict[str, Any]) -> SandboxResult:
        op_id = str(operation.get("operation_id") or "OP-UNKNOWN")
        record = {
            "operation_id": op_id,
            "executed_at": _stamp(),
            "backend": self.name,
            "dry_run": True,
            "note": "演示环境：本记录写入本地 outbox，未触碰任何真实业务系统。",
            "operation": operation,
        }
        target = self.outbox / f"{op_id}.json"
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        with (self.outbox / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "operation_id": op_id,
                        "kind": operation.get("kind"),
                        "executed_at": record["executed_at"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return SandboxResult(
            ok=True,
            backend=self.name,
            message=f"已在本地沙箱执行并落盘：{target.name}",
            payload=record,
            artifact=str(target),
        )


class DockerSandbox:
    """容器后端：无网络 + 只读根文件系统 + 仅 outbox 可写。"""

    name = "docker"
    _TEMPLATE = (
        "import json,sys\n"
        "payload=json.load(sys.stdin)\n"
        "open('/outbox/'+payload['operation_id']+'.json','w').write("
        "json.dumps(payload,ensure_ascii=False,indent=2))\n"
        "print('container executed')\n"
    )

    def __init__(self, workspace: str | Path, image: str = "python:3.12-alpine") -> None:
        self.workspace = Path(workspace)
        self.outbox = self.workspace / "outbox"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.image = image

    def probe(self) -> tuple[bool, str]:
        if shutil.which("docker") is None:
            return False, "未找到 docker 命令"
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", self.image],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:  # pragma: no cover - 环境相关
            return False, f"docker 调用失败：{exc}"
        if proc.returncode != 0:
            return False, f"本地没有镜像 {self.image}，可先执行：docker pull {self.image}"
        return True, f"容器沙箱可用（镜像 {self.image}）"

    def execute(self, operation: dict[str, Any]) -> SandboxResult:
        ok, reason = self.probe()
        if not ok:
            # fail-closed：沙箱不可用时不执行、不降级
            return SandboxResult(
                ok=False,
                backend=self.name,
                message=f"沙箱不可用，已按 fail-closed 拒绝执行（{reason}）",
                payload={"operation": operation},
            )

        op_id = str(operation.get("operation_id") or "OP-UNKNOWN")
        record = {
            "operation_id": op_id,
            "executed_at": _stamp(),
            "backend": self.name,
            "dry_run": True,
            "note": "演示环境：容器内执行，未触碰任何真实业务系统。",
            "operation": operation,
        }
        command = [
            "docker", "run", "--rm", "--network", "none",
            "--read-only", "--memory", "128m", "--cpus", "0.5", "--pids-limit", "64",
            "-v", f"{self.outbox}:/outbox",
            "-i", self.image, "python", "-c", self._TEMPLATE,
        ]
        try:
            proc = subprocess.run(
                command,
                input=json.dumps(record, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception as exc:  # pragma: no cover - 环境相关
            return SandboxResult(
                ok=False,
                backend=self.name,
                message=f"容器执行异常，已按 fail-closed 处理：{exc}",
                payload={"operation": operation},
            )

        if proc.returncode != 0:
            return SandboxResult(
                ok=False,
                backend=self.name,
                message=f"容器执行失败，已按 fail-closed 处理：{proc.stderr.strip()[:300]}",
                payload={"operation": operation},
                stdout=proc.stdout,
            )

        target = self.outbox / f"{op_id}.json"
        return SandboxResult(
            ok=True,
            backend=self.name,
            message=f"已在容器沙箱执行：{target.name}",
            payload=record,
            artifact=str(target),
            stdout=proc.stdout.strip(),
        )


def build_sandbox(name: str, workspace: str | Path) -> Sandbox:
    normalized = (name or "local").strip().lower()
    if normalized == "docker":
        return DockerSandbox(workspace)
    return LocalSandbox(workspace)


def sandbox_status(name: str, workspace: str | Path) -> tuple[bool, str]:
    return build_sandbox(name, workspace).probe()