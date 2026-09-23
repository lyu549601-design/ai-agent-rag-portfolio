"""工具层。

只读工具（自动执行）：需求解析、品类分类、规则检索、合规检查、历史检索、
价格测算、供应商推荐、合同模板匹配。

写工具（必须人工审批）：创建采购申请、下达采购订单、发送询价、修改供应商信息、
发起付款。写工具只负责"构造待执行的操作"，真正的落地一律交给沙箱，
且必须经过 policy 放行 + 人工审批。

工具注册表 ``REGISTRY`` 是白名单：Agent 只能调用注册表里的工具。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .policy import PROTECTED_SUPPLIER_FIELDS, Risk, ToolKind
from .store import ProcurementStore


class ToolError(Exception):
    """工具执行失败（参数不合法、数据不存在等），可被 Agent 记录并让模型重试。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Citation:
    kind: str
    id: str
    label: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id, "label": self.label}


@dataclass
class ToolResult:
    tool: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    citations: list[Citation] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "summary": self.summary,
            "data": self.data,
            "citations": [c.as_dict() for c in self.citations],
        }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: ToolKind
    risk: Risk
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., ToolResult]
    requires_approval: bool = False

    @property
    def is_write(self) -> bool:
        return self.kind is ToolKind.WRITE

    def json_schema(self) -> dict[str, Any]:
        """输出 OpenAI / DeepSeek 兼容的 function 描述。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --------------------------------------------------------------------------
# 品类词典 / 规格识别
# --------------------------------------------------------------------------

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "建筑钢材": ["螺纹钢", "盘螺", "高线", "线材", "圆钢", "螺纹", "HRB400", "HRB500", "HRB335", "建筑用钢"],
    "板材": ["热轧板卷", "冷轧板卷", "中厚板", "开平板", "钢板", "板卷", "热轧板", "冷轧板", "SPHC", "SPCC", "低合金板", "中板"],
    "型材": ["H型钢", "工字钢", "角钢", "槽钢", "扁钢", "方钢", "型钢"],
    "管材": ["无缝管", "焊管", "镀锌管", "螺旋管", "方管", "圆管", "钢管", "DN"],
    "不锈钢": ["不锈钢", "304", "316L", "双相钢", "耐蚀钢"],
    "优特钢/合金钢": ["合金结构钢", "模具钢", "齿轮钢", "弹簧钢", "42CrMo", "Cr12MoV", "GCr15", "20CrMnTi", "优特钢", "合金钢"],
    "耗材辅料": ["焊条", "焊丝", "切割片", "砂轮", "螺栓", "螺母", "垫片", "氧气", "乙炔", "辅料"],
    "设备备件": ["轴承", "联轴器", "减速机", "液压泵", "电机", "阀门", "密封件", "备件"],
    "加工服务": ["激光切割", "切割下料", "热处理", "镀锌加工", "表面处理", "代加工", "加工"],
    "运输服务": ["运输", "物流", "配送", "车次", "整车", "货运"],
}

HIGH_RISK_CATEGORIES = {"优特钢/合金钢"}

SPEC_PATTERNS: list[str] = [
    r"[ØΦφ]\s*\d+(?:\.\d+)?(?:\s*[×x]\s*\d+(?:\.\d+)?)*",
    r"\b(?:HRB|HPB)\s?\d{3}[A-Z]?\b",
    r"\bQ\d{3}[A-Z]?\b",
    r"\b(?:304L?|316L?|321|2205)\b",
    r"\b(?:42CrMo|Cr12MoV|GCr15|20CrMnTi|40Cr|65Mn)\b",
    r"\b(?:SPCC|SPHC|DC01|DC03)\b",
    r"\bDN\s?\d+\b",
    r"\b\d+(?:\.\d+)?\s*[×x]\s*\d+(?:\.\d+)?(?:\s*[×x]\s*\d+(?:\.\d+)?)?",
    r"\b\d+(?:\.\d+)?\s*mm\b",
    r"\bJ\d{3}\b",
    r"\bM\d+\s*[×x]\s*\d+\b",
    r"\b\d{3,4}-?2RS\b",
]

UNIT_ALIASES = {
    "t": "吨", "吨": "吨", "公吨": "吨",
    "kg": "公斤", "公斤": "公斤", "千克": "公斤",
    "米": "米", "m": "米",
    "件": "件", "个": "个", "套": "套", "片": "片", "根": "根",
    "箱": "箱", "车次": "车次", "千件": "千件", "卷": "卷", "瓶": "瓶",
    "台": "台", "辆": "辆", "桶": "桶", "盘": "盘",
}

QUANTITY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:万)?\s*(吨|公斤|千克|kg|t|米|件|个|套|片|根|箱|车次|千件|卷|瓶|台|辆|桶|盘)",
    re.IGNORECASE,
)

# 意图判定：写操作意图一律要求"明确的动作短语"，避免"分析采购是否合适"被误判成下单。
# 只有命中 WRITE_INTENTS 才会触发写工具；其余一律按只读咨询处理。
WRITE_INTENTS: list[tuple[str, list[str]]] = [
    ("payment", ["付款", "打款", "付钱", "结款", "支付货款", "发起支付", "预付货款"]),
    (
        "issue_order",
        ["下单", "下订单", "出订单", "签发订单", "直接采购", "立即采购", "立刻采购", "马上采购"],
    ),
    (
        "create_request",
        [
            "采购申请", "发起采购", "发起申请", "提申请", "申请采购", "走采购流程",
            "帮我采购", "帮我买", "创建采购", "建一个采购", "采购单",
        ],
    ),
    ("inquiry", ["询价", "发询价", "比价", "问价", "找供应商报价"]),
    (
        "update_supplier",
        [
            "修改供应商", "更新供应商", "改供应商", "变更供应商", "供应商联系人",
            "改联系人", "更新联系人", "改成", "改为",
        ],
    ),
]

QUERY_MARKERS = [
    "分析", "是否", "合适", "合理", "评估", "建议", "推荐", "比较", "对比", "查",
    "看看", "咨询", "多少", "怎么样", "能不能", "可以吗", "合规", "风险",
]

FLAG_KEYWORDS: dict[str, list[str]] = {
    "urgent": ["加急", "紧急", "尽快", "三天内", "本周内", "抢修"],
    "imported": ["进口", "日标", "JIS", "德标", "DIN", "美标", "ASTM", "欧标", "EN标准", "原装"],
    "single_source": ["单一来源", "独家", "原厂指定", "唯一供应商", "定制专供"],
    "related_party": ["关联方", "关联公司", "我朋友", "我亲属", "我们投资", "入股", "内部关联"],
    "no_material_cert": ["没有质保书", "无质保书", "不带材质单", "无材质单"],
}

SEGMENT_SPLIT_RE = re.compile(r"[，,、；;\n]+")


def _extract_spec(segment: str) -> str:
    found: list[str] = []
    for pattern in SPEC_PATTERNS:
        for match in re.finditer(pattern, segment, re.IGNORECASE):
            token = match.group(0).strip()
            if token and token not in found:
                found.append(token)
    return " ".join(found)


_SPEC_LIKE = [re.compile(pattern, re.IGNORECASE) for pattern in SPEC_PATTERNS]

NAME_STOPWORDS = [
    "帮我", "采购", "购买", "需要", "想要", "准备", "申请", "下单", "询价",
    "买", "要", "用于", "用来", "以及", "还有", "另外", "顺便", "请", "一下",
    "这批", "一批", "的", "和", "及", "各", "个", "左右", "大约", "先",
]


def _looks_like_spec(keyword: str) -> bool:
    """牌号 / 规格类词（HRB400、Q235B、304、42CrMo）不作为物料名。"""
    return any(pattern.fullmatch(keyword) for pattern in _SPEC_LIKE)


def _best_product_keyword(segment: str) -> str | None:
    best: str | None = None
    for keywords in CATEGORY_KEYWORDS.values():
        for keyword in keywords:
            if keyword.lower() in segment.lower() and not _looks_like_spec(keyword):
                if best is None or len(keyword) > len(best):
                    best = keyword
    return best


def _fallback_name(segment: str, spec: str, quantity: re.Match[str] | None) -> str:
    """没有任何词典命中时，从原始片段里剥掉数量、规格与口语词，尽量留下物料名。"""
    text = segment
    if quantity is not None:
        text = text[: quantity.start()] + " " + text[quantity.end() :]
    for token in spec.split():
        text = text.replace(token, " ")
    for word in NAME_STOPWORDS:
        text = text.replace(word, " ")
    text = re.sub(r"[0-9]+(?:\.[0-9]+)?[A-Za-z]{0,3}", " ", text)
    text = re.sub(r"[（）()\[\]【】:：,，。；;、\-—/\\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


SUPPLIER_ID_RE = re.compile(r"SUP-\d+", re.IGNORECASE)


def _supplier_id_in_text(text: str) -> str | None:
    match = SUPPLIER_ID_RE.search(text or "")
    return match.group(0).upper() if match else None


CHANGE_VALUE_RE = re.compile(r"(?:改成|改为|变更为|更新为|设为)\s*([^\s，,。；;]+)")


def _value_after_change(text: str) -> str | None:
    """从"把 X 改成 Y"里取出 Y，用于构造主数据变更参数。"""
    match = CHANGE_VALUE_RE.search(text or "")
    return match.group(1) if match else None


def _detect_intent(text: str) -> str:
    """优先识别明确的写操作意图；没有明确动作短语时按只读咨询处理。"""
    for intent, keywords in WRITE_INTENTS:
        if any(keyword in text for keyword in keywords):
            return intent
    return "query"


def _detect_flags(text: str) -> dict[str, bool]:
    return {
        key: any(keyword in text for keyword in keywords)
        for key, keywords in FLAG_KEYWORDS.items()
    }


def _parse_money(text: str) -> float | None:
    patterns = [
        r"预算\s*(\d+(?:\.\d+)?)\s*(万|万元|元)",
        r"(\d+(?:\.\d+)?)\s*万元",
        r"金额\s*(\d+(?:\.\d+)?)\s*(万|万元|元)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            unit = match.group(2) if match.lastindex and match.lastindex >= 2 else "元"
            if unit.startswith("万"):
                value *= 10000
            return value
    return None


def _parse_prepay(text: str) -> float | None:
    match = re.search(r"预付\s*(\d+(?:\.\d+)?)\s*%", text)
    return float(match.group(1)) if match else None


def _parse_project(text: str) -> str | None:
    match = re.search(r"项目[：:]\s*([^\s，,。；;]+)", text)
    if match:
        return match.group(1)
    for marker in ("用于", "供"):
        if marker in text:
            tail = text.split(marker, 1)[1]
            return tail[:20] or None
    return None


# --------------------------------------------------------------------------
# 只读工具
# --------------------------------------------------------------------------


def parse_requirement(store: ProcurementStore, text: str) -> ToolResult:
    """把自然语言采购需求解析成结构化字段（不做任何外部动作）。"""
    clean_text = (text or "").strip()
    if not clean_text:
        raise ToolError("empty_input", "需求文本为空")

    items: list[dict[str, Any]] = []
    for segment in SEGMENT_SPLIT_RE.split(clean_text):
        segment = segment.strip()
        if not segment:
            continue
        match = QUANTITY_RE.search(segment)
        if not match:
            continue
        quantity = float(match.group(1))
        unit = UNIT_ALIASES.get(match.group(2).lower(), match.group(2))
        name = _best_product_keyword(segment) or _fallback_name(
            segment, _extract_spec(segment), match
        )
        items.append(
            {
                "name": name or "未识别物料",
                "spec": _extract_spec(segment),
                "quantity": quantity,
                "unit": unit,
                "source_segment": segment,
            }
        )

    if not items:
        # 没有数量时仍允许识别"要采什么"，但数量标记为待确认
        for segment in SEGMENT_SPLIT_RE.split(clean_text):
            keyword = _best_product_keyword(segment)
            if keyword:
                items.append(
                    {
                        "name": keyword,
                        "spec": _extract_spec(segment),
                        "quantity": None,
                        "unit": None,
                        "source_segment": segment.strip(),
                    }
                )

    flags = _detect_flags(clean_text)
    requirement: dict[str, Any] = {
        "raw_text": clean_text,
        "intent": _detect_intent(clean_text),
        "items": items,
        "budget_amount": _parse_money(clean_text),
        "prepay_ratio": _parse_prepay(clean_text),
        "project": _parse_project(clean_text),
        "urgent": flags["urgent"],
        "imported": flags["imported"],
        "single_source": flags["single_source"],
        "related_party": flags["related_party"],
        "has_material_cert": not flags["no_material_cert"],
        "missing_quantity": any(item["quantity"] is None for item in items),
        "change_value": _value_after_change(clean_text),
        # 用户在原始输入里点名的供应商属于可信来源，用来与"模型自己挑的供应商"区分
        "preferred_supplier_id": _supplier_id_in_text(clean_text),
    }

    summary_parts = [
        f"{item['name']} {item['spec']}".strip()
        + (f" {item['quantity']:g}{item['unit']}" if item["quantity"] else " 数量待确认")
        for item in items
    ]
    summary = "需求解析：" + ("；".join(summary_parts) if summary_parts else "未识别到物料")
    return ToolResult(tool="parse_requirement", summary=summary, data={"requirement": requirement})


def classify_category(
    store: ProcurementStore,
    requirement: dict[str, Any] | None = None,
    text: str | None = None,
) -> ToolResult:
    """按品类词典给需求分类。离线大脑用的是关键词基线，准确率由评测集量化。"""
    if requirement is None:
        requirement = {"raw_text": text or "", "items": []}
    haystack = " ".join(
        [
            str(requirement.get("raw_text", "")),
            *[
                f"{item.get('name', '')} {item.get('spec', '')}"
                for item in requirement.get("items", [])
            ],
        ]
    )
    lowered = haystack.lower()

    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0.0
        hits: list[str] = []
        for keyword in keywords:
            if keyword.lower() in lowered:
                score += len(keyword)
                hits.append(keyword)
        if score > 0:
            scores[category] = score
            evidence[category] = hits

    if not scores:
        return ToolResult(
            tool="classify_category",
            summary="无法归类：需求中没有匹配到已知品类关键词，需要人工确认品类",
            data={
                "category": None,
                "confidence": 0.0,
                "alternatives": [],
                "evidence": [],
                "requires_human_review": True,
            },
        )

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_category, top_score = ranked[0]
    total = sum(scores.values())
    confidence = round(top_score / total, 3) if total else 0.0
    alternatives = [
        {"category": category, "score": round(score, 1)} for category, score in ranked[1:3]
    ]

    return ToolResult(
        tool="classify_category",
        summary=(
            f"品类识别为「{top_category}」（置信度 {confidence:.2f}，"
            f"证据：{'/'.join(evidence[top_category])}）"
        ),
        data={
            "category": top_category,
            "confidence": confidence,
            "alternatives": alternatives,
            "evidence": evidence[top_category],
            "requires_human_review": confidence < 0.6,
            "high_risk_category": top_category in HIGH_RISK_CATEGORIES,
        },
    )

# --------------------------------------------------------------------------
# 检索 / 合规 / 测算
# --------------------------------------------------------------------------

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def search_rules(
    store: ProcurementStore,
    query: str | None = None,
    category: str | None = None,
    limit: int = 8,
) -> ToolResult:
    """检索适用的采购规则 / 制度条款。"""
    rules = store.rules()
    tokens = [t for t in re.split(r"[，,。；;、\s]+", query or "") if len(t) >= 2]

    scored: list[tuple[float, dict[str, Any]]] = []
    for rule in rules:
        haystack = f"{rule.get('title','')}{rule.get('text','')}{rule.get('group','')}{rule.get('source','')}"
        score = float(sum(len(t) for t in tokens if t in haystack))
        trigger_type = rule.get("trigger", {}).get("type", "")
        if trigger_type == "always":
            score += 1.0
        if category and category in HIGH_RISK_CATEGORIES and trigger_type == "category_high_risk":
            score += 6.0
        if category and trigger_type.startswith("amount_"):
            score += 2.0
        scored.append((score, rule))

    scored.sort(key=lambda item: (-item[0], SEVERITY_ORDER.get(item[1].get("severity", "low"), 9)))
    selected = [rule for score, rule in scored if score > 0][:limit] or [
        rule for rule in rules if rule.get("group") == "治理"
    ][:limit]

    citations = [
        Citation(kind="rule", id=rule["id"], label=f"{rule['title']}（{rule.get('source','')}）")
        for rule in selected
    ]
    return ToolResult(
        tool="search_rules",
        summary=f"检索到 {len(selected)} 条适用规则：" + "、".join(rule["id"] for rule in selected),
        data={
            "rules": [
                {
                    "id": rule["id"],
                    "title": rule["title"],
                    "group": rule["group"],
                    "severity": rule["severity"],
                    "text": rule["text"],
                    "requires": rule.get("requires", []),
                    "source": rule.get("source", ""),
                }
                for rule in selected
            ]
        },
        citations=citations,
    )


def _compliance_context(
    store: ProcurementStore,
    requirement: dict[str, Any],
    supplier_id: str | None,
    quote: dict[str, Any] | None,
) -> dict[str, Any]:
    """汇总合规判定所需的上下文。

    金额口径优先级：实际报价金额 > 需求预算 > 报价估算 > 历史均价估算。
    口径会写进结果（amount_basis），避免把估算值当成既成事实。
    """
    quote = quote or {}
    items = requirement.get("items", [])
    category = requirement.get("category")

    reference_price = None
    reference_basis = None
    if category and items:
        try:
            reference = estimate_price(store, category=category, item=items[0].get("name") or None)
            reference_price = reference.data.get("unit_price")
            reference_basis = reference.data.get("basis")
        except ToolError:
            reference_price = None

    quoted_unit_price = quote.get("unit_price")
    unit_price = float(quoted_unit_price) if quoted_unit_price else reference_price

    estimated_amount = None
    if unit_price:
        total_qty = sum(float(item.get("quantity") or 0) for item in items)
        if total_qty:
            estimated_amount = total_qty * float(unit_price)

    amount = quote.get("amount")
    amount_basis = "实际报价"
    if amount is None:
        if requirement.get("budget_amount"):
            amount = float(requirement["budget_amount"])
            amount_basis = "需求预算"
        elif estimated_amount:
            amount = estimated_amount
            amount_basis = "报价估算" if quoted_unit_price else "历史均价估算"
        else:
            amount_basis = "无依据"

    prepay = quote.get("prepay_ratio")
    if prepay is None:
        prepay = requirement.get("prepay_ratio")

    supplier = store.supplier(supplier_id) if supplier_id else None

    deviation_pct = None
    if quoted_unit_price and reference_price:
        deviation_pct = round(
            (float(quoted_unit_price) - float(reference_price)) / float(reference_price) * 100, 2
        )

    return {
        "amount": float(amount) if amount else None,
        "amount_basis": amount_basis,
        "prepay_ratio": float(prepay) if prepay else None,
        "supplier": supplier,
        "supplier_status": (supplier or {}).get("avl_status"),
        "high_risk_category": category in HIGH_RISK_CATEGORIES,
        "deviation_pct": deviation_pct,
        "reference_price": reference_price,
        "reference_basis": reference_basis,
        "category": category,
    }


def _trigger_reasons(rule: dict[str, Any], ctx: dict[str, Any], requirement: dict[str, Any]) -> list[str]:
    trigger = rule.get("trigger", {})
    kind = trigger.get("type")
    reasons: list[str] = []

    if kind == "amount_gte":
        amount = ctx.get("amount")
        if amount is not None and amount >= float(trigger.get("threshold", 0)):
            reasons.append(
                f"本次金额 {amount:,.0f} 元 ≥ 阈值 {float(trigger['threshold']):,.0f} 元"
            )
    elif kind == "supplier_not_in_avl":
        status = ctx.get("supplier_status")
        supplier_id = (ctx.get("supplier") or {}).get("id", "")
        if status == "not_listed":
            reasons.append(f"供应商 {supplier_id} 不在合格供应商名录内（状态 {status}）")
    elif kind == "supplier_blocked":
        if ctx.get("supplier_status") == "blocked":
            reasons.append(f"供应商 {(ctx.get('supplier') or {}).get('id','')} 已被冻结")
    elif kind == "single_source":
        if requirement.get("single_source"):
            reasons.append("需求标记为单一来源采购")
    elif kind == "import_material":
        if requirement.get("imported"):
            reasons.append("需求涉及进口钢材")
    elif kind == "missing_material_cert":
        if not requirement.get("has_material_cert", True):
            reasons.append("需求声明不提供质保书 / 材质单")
    elif kind == "prepay_ratio_gt":
        prepay = ctx.get("prepay_ratio")
        if prepay is not None and prepay > float(trigger.get("threshold", 100)):
            reasons.append(f"预付比例 {prepay:g}% 超过上限 {trigger['threshold']}%")
    elif kind == "price_deviation_gt":
        deviation = ctx.get("deviation_pct")
        if deviation is not None and deviation > float(trigger.get("threshold", 10)):
            reasons.append(
                f"报价较历史均价（{ctx.get('reference_price'):,.0f} 元）高 {deviation:.2f}%"
            )
    elif kind == "category_high_risk":
        if ctx.get("high_risk_category"):
            reasons.append(f"品类「{ctx.get('category')}」属于高风险品类")
    elif kind == "related_party":
        if requirement.get("related_party"):
            reasons.append("需求声明存在关联关系")
    return reasons


def check_compliance(
    store: ProcurementStore,
    requirement: dict[str, Any],
    supplier_id: str | None = None,
    quote: dict[str, Any] | None = None,
) -> ToolResult:
    """对需求 / 供应商 / 报价做合规判定，输出适用与触发的规则。"""
    ctx = _compliance_context(store, requirement, supplier_id, quote)
    rules = store.rules()

    applicable: list[dict[str, Any]] = []
    triggered: list[dict[str, Any]] = []
    for rule in rules:
        if rule.get("trigger", {}).get("type") == "always":
            applicable.append(
                {"id": rule["id"], "title": rule["title"], "severity": rule["severity"]}
            )
            continue
        reasons = _trigger_reasons(rule, ctx, requirement)
        if reasons:
            triggered.append(
                {
                    "id": rule["id"],
                    "title": rule["title"],
                    "severity": rule["severity"],
                    "reason": "；".join(reasons),
                    "requires": rule.get("requires", []),
                    "source": rule.get("source", ""),
                }
            )

    triggered.sort(key=lambda item: SEVERITY_ORDER.get(item["severity"], 9))
    blocking = [item["id"] for item in triggered if item["severity"] in ("critical", "high")]

    citations = [
        Citation(kind="rule", id=item["id"], label=f"{item['title']}（{item.get('source','')}）")
        for item in triggered
    ] + [
        Citation(kind="rule", id=item["id"], label=f"{item['title']}（基础治理要求）")
        for item in applicable
    ]
    if ctx.get("supplier"):
        supplier = ctx["supplier"]
        citations.append(
            Citation(kind="supplier", id=supplier["id"], label=f"{supplier['name']}（{supplier['avl_status']}）")
        )

    severity_note = "无硬性阻断项" if not blocking else f"存在 {len(blocking)} 条需前置满足的要求"
    return ToolResult(
        tool="check_compliance",
        summary=f"合规检查完成：触发 {len(triggered)} 条规则，{severity_note}",
        data={
            "triggered": triggered,
            "applicable": applicable,
            "blocking_rule_ids": blocking,
            "amount_checked": ctx.get("amount"),
            "amount_basis": ctx.get("amount_basis"),
            "deviation_pct": ctx.get("deviation_pct"),
            "reference_price": ctx.get("reference_price"),
            "reference_basis": ctx.get("reference_basis"),
            "supplier_status": ctx.get("supplier_status"),
        },
        citations=citations,
    )


def _match_orders(
    store: ProcurementStore,
    category: str,
    item: str | None = None,
    spec: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    orders = [order for order in store.orders() if order.get("category") == category]
    if not orders:
        return [], "无历史数据"

    if item:
        matched = [order for order in orders if item in str(order.get("item", ""))]
        if matched:
            return matched, "同品类同物料"

    if spec:
        tokens = [token.strip() for token in spec.split() if len(token.strip()) >= 3]
        for token in tokens:
            matched = [order for order in orders if token.lower() in str(order.get("spec", "")).lower()]
            if matched:
                return matched, "同品类同规格"

    return orders, "同品类整体"


def _dominant_unit(orders: list[dict[str, Any]]) -> str | None:
    counts: dict[str, int] = {}
    for order in orders:
        unit = str(order.get("unit", ""))
        counts[unit] = counts.get(unit, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def search_history(
    store: ProcurementStore,
    category: str,
    item: str | None = None,
    spec: str | None = None,
    limit: int = 5,
) -> ToolResult:
    """检索同品类历史成交记录与统计。"""
    matched, basis = _match_orders(store, category, item, spec)
    if not matched:
        return ToolResult(
            tool="search_history",
            summary=f"品类「{category}」没有历史成交记录",
            data={"basis": basis, "count": 0, "orders": []},
        )

    unit = _dominant_unit(matched)
    same_unit = [order for order in matched if order.get("unit") == unit] or matched
    prices = [float(order["unit_price"]) for order in same_unit]
    promised = [float(order["promised_days"]) for order in same_unit if order.get("promised_days")]
    actual = [float(order["actual_days"]) for order in same_unit if order.get("actual_days")]
    incidents = [order["id"] for order in same_unit if order.get("quality_incident")]

    recent = sorted(same_unit, key=lambda order: str(order.get("date", "")), reverse=True)[:limit]
    citations = [
        Citation(
            kind="order",
            id=order["id"],
            label=f"{order['date']} {order['item']} {order['spec']} @{order['unit_price']}元/{order['unit']}",
        )
        for order in same_unit
    ]

    return ToolResult(
        tool="search_history",
        summary=(
            f"找到 {len(same_unit)} 条参考记录（{basis}，单位：{unit}）："
            f"均价 {sum(prices)/len(prices):,.0f} 元，区间 {min(prices):,.0f}–{max(prices):,.0f} 元"
        ),
        data={
            "basis": basis,
            "unit": unit,
            "count": len(same_unit),
            "avg_unit_price": round(sum(prices) / len(prices), 2),
            "min_unit_price": min(prices),
            "max_unit_price": max(prices),
            "avg_promised_days": round(sum(promised) / len(promised), 1) if promised else None,
            "avg_actual_days": round(sum(actual) / len(actual), 1) if actual else None,
            "quality_incident_orders": incidents,
            "recent_orders": [
                {
                    "id": order["id"],
                    "date": order["date"],
                    "supplier_id": order["supplier_id"],
                    "unit_price": order["unit_price"],
                    "unit": order["unit"],
                    "actual_days": order.get("actual_days"),
                }
                for order in recent
            ],
        },
        citations=citations,
    )


def estimate_price(
    store: ProcurementStore,
    category: str,
    item: str | None = None,
    spec: str | None = None,
    quantity: float | None = None,
) -> ToolResult:
    """基于历史成交做价格测算，并给出参考区间与样本量。"""
    matched, basis = _match_orders(store, category, item, spec)
    if not matched:
        raise ToolError(
            "no_reference_price",
            f"品类「{category}」缺少历史成交数据，无法给出价格参考，需人工报价",
        )

    unit = _dominant_unit(matched)
    same_unit = [order for order in matched if order.get("unit") == unit] or matched
    prices = [float(order["unit_price"]) for order in same_unit]
    avg_price = round(sum(prices) / len(prices), 2)

    citations = [
        Citation(
            kind="order",
            id=order["id"],
            label=f"{order['date']} {order['item']} {order['spec']} @{order['unit_price']}元/{order['unit']}",
        )
        for order in same_unit
    ]

    sample = len(same_unit)
    confidence = "high" if sample >= 4 else "medium" if sample >= 2 else "low"
    low = round(avg_price * 0.92, 2)
    high = round(avg_price * 1.08, 2)

    return ToolResult(
        tool="estimate_price",
        summary=(
            f"价格参考：{avg_price:,.0f} 元/{unit}（{basis}，样本 {sample} 条，"
            f"合理区间 {low:,.0f}–{high:,.0f} 元，置信度 {confidence}）"
        ),
        data={
            "unit_price": avg_price,
            "unit": unit,
            "basis": basis,
            "sample_size": sample,
            "confidence": confidence,
            "reasonable_low": low,
            "reasonable_high": high,
            "deviation_threshold_pct": 10,
            "cited_order_ids": [citation.id for citation in citations],
        },
        citations=citations,
    )


def search_suppliers(
    store: ProcurementStore,
    category: str,
    requirement: dict[str, Any] | None = None,
    limit: int = 3,
) -> ToolResult:
    """按品类筛合格供应商，返回候选名单与排除原因。"""
    requirement = requirement or {}
    eligible: list[tuple[float, dict[str, Any], str]] = []
    excluded: list[dict[str, str]] = []

    for supplier in store.suppliers():
        if category not in supplier.get("categories", []):
            continue
        status = str(supplier.get("avl_status", "unknown"))
        rating = supplier.get("rating", {})
        if status == "blocked":
            excluded.append(
                {
                    "supplier_id": supplier["id"],
                    "name": supplier["name"],
                    "rule_id": "R-SUP-002",
                    "reason": "已被冻结（R-SUP-002）",
                }
            )
            continue
        if status == "not_listed":
            excluded.append(
                {
                    "supplier_id": supplier["id"],
                    "name": supplier["name"],
                    "rule_id": "R-SUP-001",
                    "reason": "未完成准入，不在合格名录内（R-SUP-001）",
                }
            )
            continue

        score = (
            0.40 * float(rating.get("quality", 0))
            + 0.25 * float(rating.get("delivery", 0))
            + 0.20 * float(rating.get("price", 0))
            + 0.15 * float(rating.get("service", 0))
        )
        if status == "approved":
            score += 0.30
        else:
            score -= 0.30

        reason = (
            f"品类匹配、AVL {status}、综合评分 {score:.2f}、"
            f"交期 {supplier.get('lead_time_days')} 天、付款条件 {supplier.get('payment_terms')}"
        )
        if supplier.get("last_order_date"):
            reason += f"、最近合作 {supplier['last_order_date']}"
        eligible.append((score, supplier, reason))

    eligible.sort(key=lambda item: item[0], reverse=True)
    top = eligible[:limit]

    citations = [
        Citation(
            kind="supplier",
            id=supplier["id"],
            label=f"{supplier['name']}（AVL {supplier['avl_status']}，评分 {supplier.get('rating', {}).get('quality')}）",
        )
        for _score, supplier, _reason in top
    ]
    # 结论里提到的"被排除供应商"以及排除依据的规则，同样要登记为已检索，
    # 否则结论会出现"引用了没查过的记录"，引用真实率会被判为不合格。
    for entry in excluded:
        excluded_supplier = store.supplier(entry["supplier_id"])
        if excluded_supplier:
            citations.append(
                Citation(
                    kind="supplier",
                    id=excluded_supplier["id"],
                    label=f"{excluded_supplier['name']}（已排除：{entry['reason']}）",
                )
            )
        excluding_rule = store.rule(entry.get("rule_id", ""))
        if excluding_rule:
            citations.append(
                Citation(
                    kind="rule",
                    id=excluding_rule["id"],
                    label=f"{excluding_rule['title']}（排除依据）",
                )
            )

    if not top:
        summary = f"没有找到可用的合格供应商（品类「{category}」），需要人工介入"
    else:
        summary = "候选供应商：" + "、".join(supplier["id"] for _s, supplier, _r in top)

    return ToolResult(
        tool="search_suppliers",
        summary=summary,
        data={
            "candidates": [
                {
                    "supplier_id": supplier["id"],
                    "name": supplier["name"],
                    "avl_status": supplier["avl_status"],
                    "score": round(score, 2),
                    "lead_time_days": supplier.get("lead_time_days"),
                    "payment_terms": supplier.get("payment_terms"),
                    "reason": reason,
                }
                for score, supplier, reason in top
            ],
            "excluded": excluded,
        },
        citations=citations,
    )


def get_contract_template(store: ProcurementStore, category: str) -> ToolResult:
    """匹配适用的合同模板与必备条款。"""
    selected = [
        template
        for template in store.templates()
        if category in template.get("categories", []) or "*" in template.get("categories", [])
    ]
    if not selected:
        raise ToolError("no_template", f"没有匹配「{category}」的合同模板")

    if category in HIGH_RISK_CATEGORIES:
        names = {template["name"] for template in selected}
        if "质量保证协议" not in names:
            qa = store.template("CT-QA-001")
            if qa:
                selected.append(qa)

    required: list[str] = []
    for template in selected:
        for clause in template.get("required_clauses", []):
            if clause not in required:
                required.append(clause)

    citations = [
        Citation(kind="template", id=template["id"], label=f"{template['name']}（必备条款 {len(template.get('required_clauses', []))} 项）")
        for template in selected
    ]
    return ToolResult(
        tool="get_contract_template",
        summary="适用合同模板：" + "、".join(template["name"] for template in selected),
        data={
            "templates": [
                {
                    "id": template["id"],
                    "name": template["name"],
                    "key_clauses": template.get("key_clauses", []),
                    "required_clauses": template.get("required_clauses", []),
                    "note": template.get("note", ""),
                }
                for template in selected
            ],
            "required_clauses": required,
        },
        citations=citations,
    )

def get_supplier_detail(store: ProcurementStore, supplier_id: str) -> ToolResult:
    """读取单个供应商的完整档案（含备注、资质、评级）。

    备注等自由文本属于不可信内容，返回后由 Agent 统一过提示注入检测。
    """
    supplier = store.supplier(supplier_id)
    if supplier is None:
        raise ToolError("unknown_supplier", f"供应商 {supplier_id} 不存在")
    citation = Citation(
        kind="supplier",
        id=supplier["id"],
        label=f"{supplier['name']}（AVL {supplier['avl_status']}，风险等级 {supplier.get('risk_level')}）",
    )
    return ToolResult(
        tool="get_supplier_detail",
        summary=f"供应商档案 {supplier['id']}：{supplier['name']}，状态 {supplier['avl_status']}",
        data={"supplier": supplier},
        citations=[citation],
    )


def get_order_detail(store: ProcurementStore, order_id: str) -> ToolResult:
    """读取单条历史订单的完整信息（含备注，同样按不可信内容处理）。"""
    order = store.order(order_id)
    if order is None:
        raise ToolError("unknown_order", f"订单 {order_id} 不存在")
    citations = [
        Citation(
            kind="order",
            id=order["id"],
            label=f"{order['date']} {order['item']} {order['spec']} @{order['unit_price']}元/{order['unit']}",
        )
    ]
    supplier = store.supplier(str(order.get("supplier_id", "")))
    if supplier:
        citations.append(
            Citation(
                kind="supplier",
                id=supplier["id"],
                label=f"{supplier['name']}（AVL {supplier['avl_status']}）",
            )
        )
    return ToolResult(
        tool="get_order_detail",
        summary=f"订单 {order['id']}：{order['item']} {order['spec']}，供应商 {order['supplier_id']}，金额 {order['amount']:,.0f} 元",
        data={"order": order},
        citations=citations,
    )


# --------------------------------------------------------------------------
# 写工具（构造待执行操作；执行一律经由 policy + 人工审批 + 沙箱）
# --------------------------------------------------------------------------


def _operation_id(prefix: str) -> str:
    from datetime import datetime
    from uuid import uuid4

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"OP-{prefix}-{stamp}-{uuid4().hex[:4].upper()}"


def _require(args: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if args.get(key) in (None, "", [])]
    if missing:
        raise ToolError("missing_arguments", "缺少必要参数：" + "、".join(missing))


def draft_purchase_request(
    store: ProcurementStore,
    requirement_id: str | None = None,
    title: str | None = None,
    category: str | None = None,
    supplier_id: str | None = None,
    amount: float | None = None,
    items: list[dict[str, Any]] | None = None,
    reason: str | None = None,
) -> ToolResult:
    """创建采购申请草稿（内部单据，不对外发送）。"""
    _require({"category": category, "items": items}, ["category", "items"])
    items = items or []
    total = amount
    if total is None:
        total = sum(
            float(item.get("quantity") or 0) * float(item.get("unit_price") or 0)
            for item in items
        ) or None

    operation = {
        "operation_id": _operation_id("PR"),
        "kind": "create_purchase_request",
        "target_system": "purchase_request",
        "title": title or f"{category} 采购申请",
        "category": category,
        "supplier_id": supplier_id,
        "items": items,
        "amount": total,
        "currency": "CNY",
        "reason": reason,
        "requirement_id": requirement_id,
    }
    return ToolResult(
        tool="draft_purchase_request",
        summary=f"已生成采购申请草稿 {operation['operation_id']}（{category}，金额 {total or '待定'}）",
        data={"operation": operation},
    )


def issue_purchase_order(
    store: ProcurementStore,
    supplier_id: str,
    category: str,
    item: str,
    spec: str,
    quantity: float,
    unit: str,
    unit_price: float,
    amount: float,
    delivery_days: int | None = None,
    payment_terms: str | None = None,
) -> ToolResult:
    """向供应商下达采购订单（对外生效，高风险）。"""
    _require(
        {
            "supplier_id": supplier_id,
            "category": category,
            "item": item,
            "quantity": quantity,
            "unit": unit,
            "unit_price": unit_price,
            "amount": amount,
        },
        ["supplier_id", "category", "item", "quantity", "unit", "unit_price", "amount"],
    )
    supplier = store.supplier(supplier_id)
    operation = {
        "operation_id": _operation_id("PO"),
        "kind": "issue_purchase_order",
        "target_system": "purchase_order",
        "supplier_id": supplier_id,
        "supplier_name": (supplier or {}).get("name"),
        "category": category,
        "item": item,
        "spec": spec,
        "quantity": quantity,
        "unit": unit,
        "unit_price": unit_price,
        "amount": amount,
        "currency": "CNY",
        "delivery_days": delivery_days,
        "payment_terms": payment_terms or (supplier or {}).get("payment_terms"),
    }
    return ToolResult(
        tool="issue_purchase_order",
        summary=(
            f"已构造采购订单 {operation['operation_id']}：{item} {spec} "
            f"{quantity:g}{unit} × {unit_price:g} 元 = {amount:,.0f} 元"
        ),
        data={"operation": operation},
    )


def send_rfq(
    store: ProcurementStore,
    supplier_ids: list[str],
    category: str,
    item: str,
    spec: str | None = None,
    quantity: float | None = None,
    unit: str | None = None,
) -> ToolResult:
    """向候选供应商发出询价（对外发送，中风险）。"""
    _require(
        {"supplier_ids": supplier_ids, "category": category, "item": item},
        ["supplier_ids", "category", "item"],
    )
    names = []
    for supplier_id in supplier_ids:
        supplier = store.supplier(supplier_id)
        if supplier is None:
            raise ToolError("unknown_supplier", f"供应商 {supplier_id} 不存在")
        names.append(supplier["name"])

    operation = {
        "operation_id": _operation_id("RFQ"),
        "kind": "send_rfq",
        "target_system": "sourcing",
        "supplier_ids": list(supplier_ids),
        "supplier_names": names,
        "category": category,
        "item": item,
        "spec": spec,
        "quantity": quantity,
        "unit": unit,
    }
    return ToolResult(
        tool="send_rfq",
        summary=f"已构造询价单 {operation['operation_id']}，对象：" + "、".join(names),
        data={"operation": operation},
    )


def update_supplier_contact(
    store: ProcurementStore,
    supplier_id: str,
    field: str,
    value: str,
    reason: str | None = None,
) -> ToolResult:
    """修改供应商主数据（高风险，且禁止改动银行账号等敏感字段）。"""
    _require(
        {"supplier_id": supplier_id, "field": field, "value": value},
        ["supplier_id", "field", "value"],
    )
    supplier = store.supplier(supplier_id)
    if supplier is None:
        raise ToolError("unknown_supplier", f"供应商 {supplier_id} 不存在")
    if field in PROTECTED_SUPPLIER_FIELDS or "银行" in field or "账号" in field:
        raise ToolError(
            "protected_field",
            f"字段 {field} 属于受保护主数据（银行账号等），禁止由 Agent 修改，须走财务主数据流程",
        )

    operation = {
        "operation_id": _operation_id("SUP"),
        "kind": "update_supplier_contact",
        "target_system": "supplier_master",
        "supplier_id": supplier_id,
        "supplier_name": supplier.get("name"),
        "field": field,
        "value": value,
        "previous_value": supplier.get(field),
        "reason": reason,
    }
    return ToolResult(
        tool="update_supplier_contact",
        summary=f"已构造供应商主数据变更 {operation['operation_id']}：{supplier_id}.{field}",
        data={"operation": operation},
    )


def release_payment(
    store: ProcurementStore,
    order_id: str,
    amount: float,
    reason: str | None = None,
) -> ToolResult:
    """发起付款（高风险）。付款必须匹配验收单与发票（R-FIN-002）。"""
    _require({"order_id": order_id, "amount": amount}, ["order_id", "amount"])
    order = store.order(order_id)
    if order is None:
        raise ToolError("unknown_order", f"订单 {order_id} 不存在")
    if float(amount) > float(order.get("amount", 0)) * 1.02:
        raise ToolError(
            "amount_exceeds_order",
            f"付款金额 {amount:,.0f} 元超过订单金额 {order.get('amount', 0):,.0f} 元，拒绝构造付款",
        )

    operation = {
        "operation_id": _operation_id("PAY"),
        "kind": "release_payment",
        "target_system": "payment",
        "order_id": order_id,
        "supplier_id": order.get("supplier_id"),
        "amount": amount,
        "currency": "CNY",
        "reason": reason,
        "requires": ["验收单", "发票", "三单匹配"],
    }
    return ToolResult(
        tool="release_payment",
        summary=f"已构造付款指令 {operation['operation_id']}：订单 {order_id}，金额 {amount:,.0f} 元",
        data={"operation": operation},
    )


# --------------------------------------------------------------------------
# 工具注册表（白名单）
# --------------------------------------------------------------------------

REGISTRY: dict[str, ToolSpec] = {
    "parse_requirement": ToolSpec(
        name="parse_requirement",
        kind=ToolKind.READ,
        risk=Risk.NONE,
        description="把自然语言采购需求解析成结构化字段（物料、规格、数量、预算、标记）。",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "用户的原始需求描述"}},
            "required": ["text"],
        },
        handler=parse_requirement,
    ),
    "classify_category": ToolSpec(
        name="classify_category",
        kind=ToolKind.READ,
        risk=Risk.NONE,
        description="判断采购需求属于哪个品类（钢材 / 工业品品类体系）。",
        parameters={
            "type": "object",
            "properties": {
                "requirement": {"type": "object", "description": "parse_requirement 的输出"},
                "text": {"type": "string", "description": "没有结构化需求时可直接传文本"},
            },
        },
        handler=classify_category,
    ),
    "search_rules": ToolSpec(
        name="search_rules",
        kind=ToolKind.READ,
        risk=Risk.NONE,
        description="检索采购规则 / 制度条款，用于给出有依据的合规结论。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
        handler=search_rules,
    ),
    "check_compliance": ToolSpec(
        name="check_compliance",
        kind=ToolKind.READ,
        risk=Risk.NONE,
        description="对需求 / 供应商 / 报价做合规判定，返回触发与适用的规则条目。",
        parameters={
            "type": "object",
            "properties": {
                "requirement": {"type": "object"},
                "supplier_id": {"type": "string"},
                "quote": {"type": "object", "description": "报价信息：unit_price / amount / prepay_ratio 等"},
            },
            "required": ["requirement"],
        },
        handler=check_compliance,
    ),
    "search_history": ToolSpec(
        name="search_history",
        kind=ToolKind.READ,
        risk=Risk.NONE,
        description="检索同品类历史成交记录与统计（均价、交期、质量事件）。",
        parameters={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "item": {"type": "string"},
                "spec": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["category"],
        },
        handler=search_history,
    ),
    "estimate_price": ToolSpec(
        name="estimate_price",
        kind=ToolKind.READ,
        risk=Risk.NONE,
        description="基于历史成交测算参考价格与合理区间。",
        parameters={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "item": {"type": "string"},
                "spec": {"type": "string"},
                "quantity": {"type": "number"},
            },
            "required": ["category"],
        },
        handler=estimate_price,
    ),
    "search_suppliers": ToolSpec(
        name="search_suppliers",
        kind=ToolKind.READ,
        risk=Risk.NONE,
        description="按品类筛选合格供应商，返回候选名单与被排除的供应商及原因。",
        parameters={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "requirement": {"type": "object"},
                "limit": {"type": "integer"},
            },
            "required": ["category"],
        },
        handler=search_suppliers,
    ),
    "get_contract_template": ToolSpec(
        name="get_contract_template",
        kind=ToolKind.READ,
        risk=Risk.NONE,
        description="匹配适用的合同模板与必备条款。",
        parameters={
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
        },
        handler=get_contract_template,
    ),
    "get_supplier_detail": ToolSpec(
        name="get_supplier_detail",
        kind=ToolKind.READ,
        risk=Risk.NONE,
        description="读取单个供应商的完整档案（含备注与资质），用于尽调与核对。",
        parameters={
            "type": "object",
            "properties": {"supplier_id": {"type": "string"}},
            "required": ["supplier_id"],
        },
        handler=get_supplier_detail,
    ),
    "get_order_detail": ToolSpec(
        name="get_order_detail",
        kind=ToolKind.READ,
        risk=Risk.NONE,
        description="读取单条历史订单的完整信息（含备注），用于核对与溯源。",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        handler=get_order_detail,
    ),
    "draft_purchase_request": ToolSpec(
        name="draft_purchase_request",
        kind=ToolKind.WRITE,
        risk=Risk.MEDIUM,
        description="创建采购申请草稿。属于写操作，必须经过人工审批。",
        parameters={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "supplier_id": {"type": "string"},
                "amount": {"type": "number"},
                "items": {"type": "array", "items": {"type": "object"}},
                "reason": {"type": "string"},
            },
            "required": ["category", "items"],
        },
        handler=draft_purchase_request,
        requires_approval=True,
    ),
    "issue_purchase_order": ToolSpec(
        name="issue_purchase_order",
        kind=ToolKind.WRITE,
        risk=Risk.HIGH,
        description="向供应商下达采购订单。对外生效，高风险，必须经过人工审批。",
        parameters={
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string"},
                "category": {"type": "string"},
                "item": {"type": "string"},
                "spec": {"type": "string"},
                "quantity": {"type": "number"},
                "unit": {"type": "string"},
                "unit_price": {"type": "number"},
                "amount": {"type": "number"},
                "delivery_days": {"type": "integer"},
                "payment_terms": {"type": "string"},
            },
            "required": [
                "supplier_id", "category", "item", "spec",
                "quantity", "unit", "unit_price", "amount",
            ],
        },
        handler=issue_purchase_order,
        requires_approval=True,
    ),
    "send_rfq": ToolSpec(
        name="send_rfq",
        kind=ToolKind.WRITE,
        risk=Risk.MEDIUM,
        description="向候选供应商发出询价。对外发送，必须经过人工审批。",
        parameters={
            "type": "object",
            "properties": {
                "supplier_ids": {"type": "array", "items": {"type": "string"}},
                "category": {"type": "string"},
                "item": {"type": "string"},
                "spec": {"type": "string"},
                "quantity": {"type": "number"},
                "unit": {"type": "string"},
            },
            "required": ["supplier_ids", "category", "item"],
        },
        handler=send_rfq,
        requires_approval=True,
    ),
    "update_supplier_contact": ToolSpec(
        name="update_supplier_contact",
        kind=ToolKind.WRITE,
        risk=Risk.HIGH,
        description="修改供应商主数据字段。高风险，必须经过人工审批；银行账号等字段禁止修改。",
        parameters={
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string"},
                "field": {"type": "string"},
                "value": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["supplier_id", "field", "value"],
        },
        handler=update_supplier_contact,
        requires_approval=True,
    ),
    "release_payment": ToolSpec(
        name="release_payment",
        kind=ToolKind.WRITE,
        risk=Risk.HIGH,
        description="发起付款。高风险，必须经过人工审批，且需匹配验收单与发票。",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "amount"],
        },
        handler=release_payment,
        requires_approval=True,
    ),
}


def read_tool_names() -> list[str]:
    return [name for name, spec in REGISTRY.items() if spec.kind is ToolKind.READ]


def write_tool_names() -> list[str]:
    return [name for name, spec in REGISTRY.items() if spec.kind is ToolKind.WRITE]


def validate_registry() -> list[str]:
    """注册表自检：任何写工具如果没挂审批标记，视为配置错误。"""
    problems: list[str] = []
    for name, spec in REGISTRY.items():
        if spec.kind is ToolKind.WRITE and not spec.requires_approval:
            problems.append(f"写工具 {name} 未设置 requires_approval")
        if spec.kind is ToolKind.WRITE and spec.risk is Risk.NONE:
            problems.append(f"写工具 {name} 风险等级为 none，配置不合理")
        if not callable(spec.handler):
            problems.append(f"工具 {name} 没有可调用的 handler")
    return problems