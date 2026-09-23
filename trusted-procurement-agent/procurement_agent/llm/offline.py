"""离线模拟大脑。

设计目标：在没有 API Key 的情况下，把"需求识别 -> 分类 -> 查规则/历史/供应商 ->
合规判断 -> 给建议 -> 写操作人工审批"整条链路完整跑通，且结果可复现。

它不是大模型，而是一个规则驱动的确定性计划器：

* 每一步调哪个工具，由已经拿到的观察结果决定；
* 最终结论全部由工具返回的真实数据拼装，并带上真实记录编号；
* 计划里没有任何"绕过审批"的分支 —— 写操作一律交给 Agent 的门禁处理。

换成 DeepSeek 后工具循环完全一致，因此评测可以量化"换模型带来的提升"。
"""

from __future__ import annotations

import re
from typing import Any

from .base import Action, FinalAnswer, ToolCall

SUPPLIER_ID_RE = re.compile(r"SUP-\d+", re.IGNORECASE)
ORDER_ID_RE = re.compile(r"PO-\d{4}-\d+", re.IGNORECASE)


def _seen(observations: list[dict[str, Any]], tool: str) -> bool:
    return any(item.get("tool") == tool for item in observations)


def _count(observations: list[dict[str, Any]], tool: str) -> int:
    return sum(1 for item in observations if item.get("tool") == tool)


def _data(observations: list[dict[str, Any]], tool: str) -> dict[str, Any]:
    for item in observations:
        if item.get("tool") == tool and item.get("status") == "ok":
            return item.get("data") or {}
    return {}


def _supplier_in_text(text: str) -> str | None:
    match = SUPPLIER_ID_RE.search(text or "")
    return match.group(0).upper() if match else None


def _order_in_text(text: str) -> str | None:
    match = ORDER_ID_RE.search(text or "")
    return match.group(0).upper() if match else None


class OfflineBrain:
    """规则驱动的确定性计划器，充当"离线模拟大脑"。"""

    name = "offline"

    def next_action(
        self,
        *,
        user_input: str,
        observations: list[dict[str, Any]],
        tools: dict[str, Any],
    ) -> Action:
        # 1. 需求识别
        if not _seen(observations, "parse_requirement"):
            return ToolCall("parse_requirement", {"text": user_input})
        requirement = _data(observations, "parse_requirement").get("requirement") or {}

        # 2. 品类分类
        if not _seen(observations, "classify_category"):
            return ToolCall("classify_category", {"requirement": requirement})
        classification = _data(observations, "classify_category")
        category = classification.get("category")

        items = requirement.get("items") or []
        first = items[0] if items else {}
        intent = str(requirement.get("intent") or "query")
        supplier_hint = _supplier_in_text(user_input)
        order_hint = _order_in_text(user_input)

        # 3. 规则检索（品类没识别出来也先查，便于给出依据）
        if not _seen(observations, "search_rules"):
            return ToolCall(
                "search_rules", {"query": user_input, "category": category or None}
            )

        # 4. 付款类：订单自带品类与金额，不依赖物料品类识别
        if intent == "payment":
            return self._payment_flow(
                requirement=requirement,
                observations=observations,
                order_hint=order_hint,
                classification=classification,
            )

        # 5. 供应商主数据变更：以供应商档案为准，与物料品类无关
        if intent == "update_supplier":
            return self._supplier_update_flow(
                requirement=requirement,
                observations=observations,
                supplier_hint=supplier_hint,
                classification=classification,
            )

        # 6. 品类识别不出来 -> 停下交人工确认，不猜
        if not category:
            return FinalAnswer(
                text=self._unclassified_text(requirement, observations),
                citation_ids=self._citation_ids(observations),
            )

        # 6. 价格测算 + 历史成交
        if not _seen(observations, "estimate_price"):
            return ToolCall(
                "estimate_price",
                {"category": category, "item": first.get("name"), "spec": first.get("spec")},
            )
        if not _seen(observations, "search_history"):
            return ToolCall(
                "search_history",
                {"category": category, "item": first.get("name"), "spec": first.get("spec")},
            )

        # 7. 供应商检索
        if not _seen(observations, "search_suppliers"):
            return ToolCall(
                "search_suppliers", {"category": category, "requirement": requirement}
            )
        candidates = _data(observations, "search_suppliers").get("candidates") or []
        target_supplier = supplier_hint or (candidates[0]["supplier_id"] if candidates else None)

        # 8. 供应商尽调（档案备注属于不可信内容）
        if target_supplier and not _seen(observations, "get_supplier_detail"):
            return ToolCall("get_supplier_detail", {"supplier_id": target_supplier})

        # 9/10. 合规检查：先按需求本身，再带上首选供应商
        checks = _count(observations, "check_compliance")
        if checks == 0:
            return ToolCall("check_compliance", {"requirement": requirement})
        if checks == 1 and target_supplier:
            return ToolCall(
                "check_compliance",
                {"requirement": requirement, "supplier_id": target_supplier},
            )

        # 11. 合同模板
        if not _seen(observations, "get_contract_template"):
            return ToolCall("get_contract_template", {"category": category})

        # 12. 写操作（需要落地才发起，且必然走审批）
        planned = self._plan_write(
            intent=intent,
            requirement=requirement,
            category=category,
            observations=observations,
            target_supplier=target_supplier,
            candidates=candidates,
            supplier_hint=supplier_hint,
        )
        if planned and not _seen(observations, planned.name):
            return planned

        # 13. 结论
        return FinalAnswer(
            text=self._compose(
                requirement=requirement,
                classification=classification,
                observations=observations,
                category=category,
                target_supplier=target_supplier,
            ),
            citation_ids=self._citation_ids(observations),
        )

    # ------------------------------------------------------------------ 付款
    def _payment_flow(
        self,
        *,
        requirement: dict[str, Any],
        observations: list[dict[str, Any]],
        order_hint: str | None,
        classification: dict[str, Any],
    ) -> Action:
        if order_hint and not _seen(observations, "get_order_detail"):
            return ToolCall("get_order_detail", {"order_id": order_hint})

        order = _data(observations, "get_order_detail").get("order") or {}

        if not _seen(observations, "check_compliance"):
            quote = {"amount": float(order.get("amount") or 0)} if order else None
            return ToolCall("check_compliance", {"requirement": requirement, "quote": quote})

        if order_hint and order and not _seen(observations, "release_payment"):
            return ToolCall(
                "release_payment",
                {
                    "order_id": order_hint,
                    "amount": float(order.get("amount") or 0),
                    "reason": "按订单金额发起付款",
                },
            )

        return FinalAnswer(
            text=self._compose(
                requirement=requirement,
                classification=classification,
                observations=observations,
                category=classification.get("category"),
                target_supplier=None,
            ),
            citation_ids=self._citation_ids(observations),
        )

    # ------------------------------------------------------ 供应商主数据变更
    def _supplier_update_flow(
        self,
        *,
        requirement: dict[str, Any],
        observations: list[dict[str, Any]],
        supplier_hint: str | None,
        classification: dict[str, Any],
    ) -> Action:
        if not supplier_hint:
            return FinalAnswer(
                text=(
                    "【需求识别】未指定供应商编号\n"
                    "【结论】修改供应商主数据必须明确到具体供应商（例如 SUP-001），"
                    "否则不发起任何写操作。请补充供应商编号后重试。"
                ),
                citation_ids=self._citation_ids(observations),
            )

        if not _seen(observations, "get_supplier_detail"):
            return ToolCall("get_supplier_detail", {"supplier_id": supplier_hint})

        supplier = _data(observations, "get_supplier_detail").get("supplier") or {}
        categories = supplier.get("categories") or []
        if categories and not requirement.get("category"):
            requirement["category"] = categories[0]

        if not _seen(observations, "check_compliance"):
            return ToolCall(
                "check_compliance",
                {"requirement": requirement, "supplier_id": supplier_hint},
            )

        if not _seen(observations, "update_supplier_contact"):
            return ToolCall(
                "update_supplier_contact",
                {
                    "supplier_id": supplier_hint,
                    "field": "contact_person",
                    "value": requirement.get("change_value") or "按需求更新",
                    "reason": "按采购需求更新供应商联系人（写操作，需人工审批）",
                },
            )

        return FinalAnswer(
            text=self._compose(
                requirement=requirement,
                classification=classification,
                observations=observations,
                category=requirement.get("category"),
                target_supplier=supplier_hint,
            ),
            citation_ids=self._citation_ids(observations),
        )

    # ------------------------------------------------------------------ 写操作
    def _plan_write(
        self,
        *,
        intent: str,
        requirement: dict[str, Any],
        category: str,
        observations: list[dict[str, Any]],
        target_supplier: str | None,
        candidates: list[dict[str, Any]],
        supplier_hint: str | None,
    ) -> ToolCall | None:
        items = requirement.get("items") or []
        first = items[0] if items else {}
        price = _data(observations, "estimate_price").get("unit_price")
        quantity = first.get("quantity")
        unit = first.get("unit")
        amount = None
        if price and quantity:
            amount = round(float(price) * float(quantity), 2)

        if intent == "issue_order":
            if not (target_supplier and quantity and price):
                return None
            return ToolCall(
                "issue_purchase_order",
                {
                    "supplier_id": target_supplier,
                    "category": category,
                    "item": first.get("name") or "",
                    "spec": first.get("spec") or "",
                    "quantity": float(quantity),
                    "unit": unit or "吨",
                    "unit_price": float(price),
                    "amount": float(amount),
                },
            )

        if intent == "create_request":
            payload_items = [
                {
                    "name": item.get("name"),
                    "spec": item.get("spec"),
                    "quantity": item.get("quantity"),
                    "unit": item.get("unit"),
                    "unit_price": price,
                }
                for item in items
            ]
            return ToolCall(
                "draft_purchase_request",
                {
                    "category": category,
                    "supplier_id": target_supplier,
                    "amount": float(amount) if amount else requirement.get("budget_amount"),
                    "items": payload_items,
                    "reason": "按需求自动生成采购申请草稿",
                },
            )

        if intent == "inquiry":
            supplier_ids = [supplier_hint] if supplier_hint else [
                candidate["supplier_id"] for candidate in candidates[:3]
            ]
            if not supplier_ids:
                return None
            return ToolCall(
                "send_rfq",
                {
                    "supplier_ids": supplier_ids,
                    "category": category,
                    "item": first.get("name") or "",
                    "spec": first.get("spec") or "",
                    "quantity": quantity,
                    "unit": unit,
                },
            )

        return None

    # ------------------------------------------------------------------ 文案
    @staticmethod
    def _citation_ids(observations: list[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        for item in observations:
            for citation in item.get("citations", []):
                if citation.get("id") not in ids:
                    ids.append(citation["id"])
        return ids

    @staticmethod
    def _write_outcome(observations: list[dict[str, Any]], tool: str) -> str | None:
        for item in reversed(observations):
            if item.get("tool") != tool:
                continue
            status = item.get("status")
            if status == "ok":
                return f"已执行（{item.get('summary', '')}）"
            if status == "blocked":
                return f"已被拦截，未执行（{item.get('summary', '')}）"
            if status == "error":
                return f"执行失败（{item.get('summary', '')}）"
        return None

    def _unclassified_text(
        self, requirement: dict[str, Any], observations: list[dict[str, Any]]
    ) -> str:
        items = requirement.get("items") or []
        described = (
            "、".join(f"{item.get('name')} {item.get('spec')}".strip() for item in items)
            or "未识别到明确物料"
        )
        rules = _data(observations, "search_rules").get("rules") or []
        rule_text = "、".join(rule["id"] for rule in rules) or "无"
        return (
            f"【需求识别】{described}\n"
            "【品类判定】无法从现有品类词典中确定归属（置信度过低）。按治理要求不猜测，"
            "需要人工确认品类后再继续，也不发起任何写操作。\n"
            f"【规则检索】{rule_text}\n"
            "【下一步建议】由采购经办确认品类，或提供物料编码 / 图纸号后重试。"
        )

    def _compose(
        self,
        *,
        requirement: dict[str, Any],
        classification: dict[str, Any],
        observations: list[dict[str, Any]],
        category: str | None,
        target_supplier: str | None,
    ) -> str:
        items = requirement.get("items") or []
        lines: list[str] = []

        described = (
            "、".join(
                f"{item.get('name')} {item.get('spec')}".strip()
                + (f" {item['quantity']:g}{item['unit']}" if item.get("quantity") else " 数量待确认")
                for item in items
            )
            or "未识别到明确物料（按订单号处理）"
        )
        lines.append(f"【需求识别】{described}")

        if category:
            lines.append(
                f"【品类判定】{category}（置信度 {classification.get('confidence', 0):.2f}）"
            )

        order = _data(observations, "get_order_detail").get("order")
        if order:
            lines.append(
                f"【关联订单】{order['id']}｜{order['item']} {order['spec']}｜"
                f"供应商 {order['supplier_id']}｜金额 {order['amount']:,.0f} 元｜"
                f"付款条件 {order.get('payment_terms', '-')}"
            )

        rules = _data(observations, "search_rules").get("rules") or []
        if rules:
            lines.append(
                "【规则检索】"
                + "、".join(f"{rule['id']} {rule['title']}" for rule in rules[:3])
                + "（是否触发见下方合规结论）"
            )

        history = _data(observations, "search_history")
        if history.get("count"):
            lines.append(
                f"【历史参考】{history['count']} 条同一口径记录，均价 "
                f"{history['avg_unit_price']:,.0f} 元/{history.get('unit', '')}，"
                f"区间 {history['min_unit_price']:,.0f}–{history['max_unit_price']:,.0f} 元"
            )

        estimate = _data(observations, "estimate_price")
        if estimate.get("unit_price"):
            lines.append(
                f"【价格参考】{estimate['unit_price']:,.0f} 元/{estimate.get('unit', '吨')}"
                f"（口径：{estimate.get('basis')}，样本 {estimate.get('sample_size')} 条，"
                f"合理区间 {estimate['reasonable_low']:,.0f}–{estimate['reasonable_high']:,.0f} 元）"
                f" 依据 {'、'.join(estimate.get('cited_order_ids', [])[:4])}"
            )

        suppliers = _data(observations, "search_suppliers")
        candidates = suppliers.get("candidates") or []
        if candidates:
            lines.append(
                "【候选供应商】"
                + "；".join(
                    f"{c['supplier_id']} {c['name']}（评分 {c['score']}，交期 {c['lead_time_days']} 天）"
                    for c in candidates[:3]
                )
            )
        for excluded in (suppliers.get("excluded") or [])[:3]:
            lines.append(
                f"【已排除】{excluded['supplier_id']} {excluded['name']}：{excluded['reason']}"
            )

        compliance = _data(observations, "check_compliance")
        triggered = compliance.get("triggered") or []
        amount = compliance.get("amount_checked")
        if triggered:
            lines.append("【合规结论】")
            for rule in triggered:
                lines.append(f"  - {rule['id']}（{rule['severity']}）{rule['reason']}")
            if amount:
                lines.append(f"  金额口径：{amount:,.0f} 元（{compliance.get('amount_basis')}）")
        else:
            lines.append("【合规结论】未触发限制性条款，按基础治理要求执行")

        template = _data(observations, "get_contract_template")
        if template.get("templates"):
            lines.append(
                "【合同与质量】适用模板："
                + "、".join(t["name"] for t in template["templates"][:4])
            )
            lines.append(
                "  必备条款：" + "、".join(template.get("required_clauses", [])[:6]) + " 等"
            )

        if target_supplier:
            lines.append(f"【首选供应商】{target_supplier}")

        write_tools = (
            "draft_purchase_request",
            "issue_purchase_order",
            "send_rfq",
            "update_supplier_contact",
            "release_payment",
        )
        for tool in write_tools:
            outcome = self._write_outcome(observations, tool)
            if outcome:
                lines.append(f"【写操作】{tool}：{outcome}")

        if not any(_seen(observations, tool) for tool in write_tools):
            lines.append("【建议动作】以上为只读分析结论，未执行任何写操作；如需落地请明确指令。")

        lines.append(
            "【治理说明】只读查询自动执行；任何写 / 改操作都需要人工审批后才能落地（R-GOV-001）。"
        )
        return "\n".join(lines)