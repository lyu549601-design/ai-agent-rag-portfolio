"""数据访问层。

演示版从本地 JSON 文件读取脱敏样例数据；接口用 Protocol 定义，
以后接真实系统（ERP / SRM / 合同系统）时，只要实现同一套方法即可，
Agent 与工具层代码不需要改动。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

class ProcurementStore(Protocol):
    """数据访问接口（可替换为真实系统）。"""

    def rules(self) -> list[dict[str, Any]]: ...

    def rule(self, rule_id: str) -> dict[str, Any] | None: ...

    def orders(self) -> list[dict[str, Any]]: ...

    def order(self, order_id: str) -> dict[str, Any] | None: ...

    def suppliers(self) -> list[dict[str, Any]]: ...

    def supplier(self, supplier_id: str) -> dict[str, Any] | None: ...

    def templates(self) -> list[dict[str, Any]]: ...

    def template(self, template_id: str) -> dict[str, Any] | None: ...

    def dataset_version(self) -> str: ...

    def resolve(self, ref_id: str) -> tuple[str, dict[str, Any]] | None: ...


class DataError(RuntimeError):
    """数据文件缺失或格式错误。"""


class JsonStore:
    """本地 JSON 数据源（演示 / 离线评测用）。"""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._cache: dict[str, dict[str, Any]] = {}
        self._load()

    # ---------------------------------------------------------------- loading
    def _load(self) -> None:
        files = {
            "rules": ("rules.json", "rules"),
            "suppliers": ("suppliers.json", "suppliers"),
            "orders": ("history_orders.json", "orders"),
            "templates": ("contract_templates.json", "templates"),
        }
        for key, (filename, list_key) in files.items():
            path = self.data_dir / filename
            if not path.exists():
                raise DataError(f"缺少数据文件：{path}")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise DataError(f"数据文件不是合法 JSON：{path}（{exc}）") from exc
            if not isinstance(payload, dict) or list_key not in payload:
                raise DataError(f"数据文件结构不符合预期：{path}")
            self._cache[key] = payload

        self._index: dict[str, tuple[str, dict[str, Any]]] = {}
        for kind, key in (
            ("rule", "rules"),
            ("supplier", "suppliers"),
            ("order", "orders"),
            ("template", "templates"),
        ):
            for record in self._cache[key][key]:
                record_id = str(record.get("id", ""))
                if record_id:
                    self._index[record_id.upper()] = (kind, record)

    # ----------------------------------------------------------------- access
    def rules(self) -> list[dict[str, Any]]:
        return list(self._cache["rules"]["rules"])

    def rule(self, rule_id: str) -> dict[str, Any] | None:
        found = self._index.get(str(rule_id).upper())
        return found[1] if found and found[0] == "rule" else None

    def suppliers(self) -> list[dict[str, Any]]:
        return list(self._cache["suppliers"]["suppliers"])

    def supplier(self, supplier_id: str) -> dict[str, Any] | None:
        found = self._index.get(str(supplier_id).upper())
        return found[1] if found and found[0] == "supplier" else None

    def orders(self) -> list[dict[str, Any]]:
        return list(self._cache["orders"]["orders"])

    def order(self, order_id: str) -> dict[str, Any] | None:
        found = self._index.get(str(order_id).upper())
        return found[1] if found and found[0] == "order" else None

    def templates(self) -> list[dict[str, Any]]:
        return list(self._cache["templates"]["templates"])

    def template(self, template_id: str) -> dict[str, Any] | None:
        found = self._index.get(str(template_id).upper())
        return found[1] if found and found[0] == "template" else None

    def dataset_version(self) -> str:
        return str(self._cache["rules"].get("version", "unknown"))

    # ------------------------------------------------------------------ utils
    def resolve(self, ref_id: str) -> tuple[str, dict[str, Any]] | None:
        """按编号解析任意记录，返回 (kind, record)。用于校验引用是否真实存在。"""
        return self._index.get(str(ref_id).upper())

    def stats(self) -> dict[str, int]:
        return {
            "rules": len(self.rules()),
            "suppliers": len(self.suppliers()),
            "orders": len(self.orders()),
            "templates": len(self.templates()),
        }

    def label(self, ref_id: str) -> str:
        """给引用编号生成一个可读标签，例如 "R-GOV-001 写操作必须人工审批"。"""
        found = self.resolve(ref_id)
        if not found:
            return str(ref_id)
        kind, record = found
        title = record.get("title") or record.get("name") or record.get("item") or ""
        return f"{ref_id} {title}".strip()