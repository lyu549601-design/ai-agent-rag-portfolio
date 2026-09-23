"""Hash-chain 防篡改审计日志。

每条记录包含前一条的哈希，任何一条被改动都会导致整条链校验失败。
演示版写本地 JSONL 文件；接真实系统时可换成 WORM 存储 / 数据库表。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64


def canonical_json(payload: Any) -> str:
    """稳定的 JSON 序列化，保证同样的内容得到同样的哈希。"""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(payload: Any) -> str:
    """对任意内容取摘要，用于记录工具参数 / 结果的指纹。"""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class AuditRecord:
    seq: int
    ts: str
    actor: str
    event: str
    data: dict[str, Any]
    prev_hash: str
    hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "actor": self.actor,
            "event": self.event,
            "data": self.data,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    checked: int
    total: int
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "total": self.total,
            "message": self.message,
        }


class AuditLog:
    """追加写入 + 可离线校验的审计日志。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- writing
    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _last(self) -> dict[str, Any] | None:
        records = self.records()
        return records[-1] if records else None

    def append(self, actor: str, event: str, **data: Any) -> AuditRecord:
        last = self._last()
        seq = int(last["seq"]) + 1 if last else 1
        prev_hash = str(last["hash"]) if last else GENESIS_HASH

        body: dict[str, Any] = {
            "seq": seq,
            "ts": _now(),
            "actor": actor,
            "event": event,
            "data": data,
            "prev_hash": prev_hash,
        }
        chain_hash = hashlib.sha256(
            f"{prev_hash}|{canonical_json(body)}".encode("utf-8")
        ).hexdigest()

        record = AuditRecord(
            seq=seq,
            ts=body["ts"],
            actor=actor,
            event=event,
            data=data,
            prev_hash=prev_hash,
            hash=chain_hash,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record.as_dict()) + "\n")
        return record

    # -------------------------------------------------------------- verifying
    def verify(self) -> VerificationResult:
        records = self.records()
        if not records:
            return VerificationResult(True, 0, 0, "审计日志为空，视为通过")

        prev_hash = GENESIS_HASH
        for index, raw in enumerate(records, start=1):
            body = {
                "seq": raw.get("seq"),
                "ts": raw.get("ts"),
                "actor": raw.get("actor"),
                "event": raw.get("event"),
                "data": raw.get("data"),
                "prev_hash": raw.get("prev_hash"),
            }
            if raw.get("prev_hash") != prev_hash:
                return VerificationResult(
                    False,
                    index - 1,
                    len(records),
                    f"第 {raw.get('seq')} 条记录的 prev_hash 与前一条不一致",
                )
            expect = hashlib.sha256(
                f"{prev_hash}|{canonical_json(body)}".encode("utf-8")
            ).hexdigest()
            if raw.get("hash") != expect:
                return VerificationResult(
                    False,
                    index - 1,
                    len(records),
                    f"第 {raw.get('seq')} 条记录的哈希校验失败（内容已被修改）",
                )
            prev_hash = str(raw.get("hash"))

        return VerificationResult(True, len(records), len(records), "审计链完整，未发现篡改")

    def tail(self, count: int = 10) -> list[dict[str, Any]]:
        return self.records()[-count:]