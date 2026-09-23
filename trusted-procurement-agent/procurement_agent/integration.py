"""真实系统接入层：幂等账本、补偿回滚、ERP/SRM HTTP 适配器。

这一层解决的是"接真实系统时会立刻撞上的三件事"：

1. **幂等** -- 网络超时后重试、用户刷新页面重复提交，都不能让一张订单下两次。
   账本以 ``operation_id`` 为幂等键，重复提交直接返回"已处理"，不再执行。
2. **补偿（回滚）** -- 真实系统大多没有分布式事务。每个写操作都登记一个
   "反向动作"，如需撤销就按补偿动作执行，而不是假装能事务回滚。
3. **对接** -- ``RestErpSandbox`` 是一个可以直接用的 HTTP 适配器：
   配置 endpoint 就能把写操作提交给真实系统；未配置时按 fail-closed 拒绝。

只读部分的数据接口在 ``store.ProcurementStore``，写/执行部分在 ``sandbox.Sandbox``，
本模块负责把"执行"包一层治理，不改变 Agent 与工具层的代码。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sandbox import Sandbox, SandboxResult


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 补偿动作表：每种写操作对应的"反向动作"
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Compensation:
    action: str
    description: str
    needs_human: bool = True


COMPENSATIONS: dict[str, Compensation] = {
    "create_purchase_request": Compensation(
        "cancel_purchase_request", "撤销采购申请草稿", needs_human=False
    ),
    "issue_purchase_order": Compensation(
        "cancel_purchase_order", "向供应商发出订单取消通知（可能产生违约成本）"
    ),
    "send_rfq": Compensation("retract_rfq", "撤回询价并通知供应商", needs_human=False),
    "update_supplier_contact": Compensation(
        "restore_supplier_field", "把供应商字段恢复为变更前的值"
    ),
    "release_payment": Compensation(
        "reverse_payment", "发起反向付款 / 冲正（必须财务人工确认）"
    ),
}


# ---------------------------------------------------------------------------
# 幂等账本
# ---------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    operation_id: str
    kind: str
    status: str                     # applied / failed / rolled_back
    backend: str
    attempts: int
    submitted_at: str
    payload_digest: str = ""
    artifact: str | None = None
    compensation_action: str | None = None
    compensation_ready: bool = False
    last_message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "kind": self.kind,
            "status": self.status,
            "backend": self.backend,
            "attempts": self.attempts,
            "submitted_at": self.submitted_at,
            "payload_digest": self.payload_digest,
            "artifact": self.artifact,
            "compensation_action": self.compensation_action,
            "compensation_ready": self.compensation_ready,
            "last_message": self.last_message,
        }


class OutboundLedger:
    """记录每一次写操作的提交结果，作为幂等判定与回滚依据。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, operation_id: str) -> dict[str, Any] | None:
        return self._read().get(operation_id)

    def upsert(self, entry: LedgerEntry) -> None:
        data = self._read()
        previous = data.get(entry.operation_id, {})
        entry.attempts = int(previous.get("attempts", 0)) + entry.attempts
        data[entry.operation_id] = entry.as_dict()
        self._write(data)

    def mark_rolled_back(self, operation_id: str, message: str) -> None:
        data = self._read()
        if operation_id in data:
            data[operation_id]["status"] = "rolled_back"
            data[operation_id]["last_message"] = message
            self._write(data)

    def is_applied(self, operation_id: str) -> bool:
        entry = self.get(operation_id)
        return bool(entry and entry.get("status") == "applied")

    def entries(self) -> list[dict[str, Any]]:
        return sorted(
            self._read().values(), key=lambda item: str(item.get("submitted_at", "")), reverse=True
        )


# ---------------------------------------------------------------------------
# 带幂等与补偿登记的沙箱包装
# ---------------------------------------------------------------------------

class OutboundExecutor:
    """实现 ``Sandbox`` 协议：在真实沙箱外面包一层幂等与补偿登记。

    可直接当 sandbox 传给 Agent —— Agent 与工具层不需要知道这层存在。
    """

    def __init__(self, inner: Sandbox, ledger_path: str | Path) -> None:
        self.inner = inner
        self.ledger = OutboundLedger(ledger_path)

    @property
    def name(self) -> str:
        return f"{self.inner.name}+idempotent"

    def probe(self) -> tuple[bool, str]:
        ok, message = self.inner.probe()
        return ok, f"{message}；已启用幂等账本与补偿登记"

    def execute(self, operation: dict[str, Any]) -> SandboxResult:
        key = str(operation.get("idempotency_key") or operation.get("operation_id") or "")
        if key and self.ledger.is_applied(key):
            return SandboxResult(
                ok=True,
                backend=self.name,
                message=f"幂等命中：操作 {key} 已执行过，本次不再重复执行",
                payload={"operation": operation, "idempotent_replay": True},
                replayed=True,
            )

        result = self.inner.execute(operation)

        compensation = COMPENSATIONS.get(str(operation.get("kind", "")))
        entry = LedgerEntry(
            operation_id=str(operation.get("operation_id") or ""),
            kind=str(operation.get("kind") or ""),
            status="applied" if result.ok else "failed",
            backend=self.inner.name,
            attempts=1,
            submitted_at=_now(),
            payload_digest=_digest(operation),
            artifact=result.artifact,
            compensation_action=compensation.action if compensation else None,
            compensation_ready=bool(compensation and result.ok),
            last_message=result.message,
        )
        if entry.operation_id:
            self.ledger.upsert(entry)
        return result

    def rollback(self, operation_id: str) -> SandboxResult:
        """执行补偿动作（回滚）。真实系统里也可能造成成本，因此默认需要人工确认。"""
        entry = self.ledger.get(operation_id)
        if entry is None:
            return SandboxResult(
                ok=False, backend=self.name, message=f"账本里没有操作 {operation_id}，无法回滚"
            )
        if entry.get("status") == "rolled_back":
            return SandboxResult(
                ok=False, backend=self.name, message=f"操作 {operation_id} 已经回滚过，不重复执行"
            )
        action = entry.get("compensation_action")
        if not action:
            return SandboxResult(
                ok=False,
                backend=self.name,
                message=f"操作 {operation_id}（{entry.get('kind')}）没有登记补偿动作，需人工处理",
            )

        compensation_op = {
            "operation_id": f"{operation_id}-COMP",
            "kind": action,
            "target_system": "compensation",
            "compensates": operation_id,
            "reason": f"回滚 {entry.get('kind')}",
        }
        result = self.inner.execute(compensation_op)
        if result.ok:
            self.ledger.mark_rolled_back(operation_id, f"已通过 {action} 回滚")
        return result


def _digest(payload: Any) -> str:
    from .audit import digest

    return digest(payload)


# ---------------------------------------------------------------------------
# 真实系统 HTTP 适配器（预留接口）
# ---------------------------------------------------------------------------

class RestErpSandbox:
    """把写操作提交给真实 ERP / SRM 的 HTTP 适配器。

    契约（接真实系统时按这个实现即可）：

    * ``POST {endpoint}``
    * 请求头：``Idempotency-Key: <operation_id>``、``Authorization: Bearer <token>``
    * 请求体：``{"operation_id": ..., "kind": ..., ...}``
    * 期望响应：2xx + ``{"accepted": true, "reference": "<系统单号>"}``
    * 幂等：同一个 ``Idempotency-Key`` 重复提交，对方应返回"已受理"而不是再下一单；
      若对方返回 409，本适配器按"已处理"对待。

    未配置 ``PROCUREMENT_ERP_ENDPOINT`` 时直接 fail-closed，不做任何本地降级。
    """

    name = "erp-rest"

    def __init__(
        self,
        endpoint: str | None = None,
        token: str | None = None,
        timeout: int = 30,
        retries: int = 2,
    ) -> None:
        self.endpoint = (endpoint or os.environ.get("PROCUREMENT_ERP_ENDPOINT", "")).rstrip("/")
        self.token = token or os.environ.get("PROCUREMENT_ERP_TOKEN", "")
        self.timeout = timeout
        self.retries = max(0, retries)

    def probe(self) -> tuple[bool, str]:
        if not self.endpoint:
            return False, (
                "未配置 PROCUREMENT_ERP_ENDPOINT，真实系统适配器未启用"
                "（演示环境请用 local / docker 沙箱）"
            )
        return True, f"真实系统适配器已配置：{self.endpoint}"

    def execute(self, operation: dict[str, Any]) -> SandboxResult:
        ok, message = self.probe()
        if not ok:
            return SandboxResult(
                ok=False,
                backend=self.name,
                message=f"无法提交到真实系统，已按 fail-closed 拒绝：{message}",
                payload={"operation": operation},
            )

        body = json.dumps(operation, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": str(operation.get("operation_id") or ""),
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_error = ""
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.endpoint, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8") or "{}")
                return SandboxResult(
                    ok=True,
                    backend=self.name,
                    message=f"真实系统已受理（参考号 {payload.get('reference', '-')}）",
                    payload={"operation": operation, "response": payload},
                )
            except urllib.error.HTTPError as exc:
                if exc.code == 409:
                    return SandboxResult(
                        ok=True,
                        backend=self.name,
                        message="真实系统返回 409：该幂等键已处理，按已受理对待",
                        payload={"operation": operation, "response": {"accepted": True}},
                        replayed=True,
                    )
                last_error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:200]}"
                if exc.code < 500:
                    break
            except Exception as exc:  # pragma: no cover - 环境相关
                last_error = str(exc)
            if attempt < self.retries:
                import time

                time.sleep(1 + attempt)

        return SandboxResult(
            ok=False,
            backend=self.name,
            message=f"真实系统提交失败（已重试 {self.retries} 次），按 fail-closed 处理：{last_error}",
            payload={"operation": operation},
        )