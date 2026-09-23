"""治理策略层。

三件事：

1. 风险分级 -- 每个工具声明只读 / 写以及风险等级；
2. 审批门禁 -- 写操作默认必须人工审批，且高风险需要更强确认；
3. 来源绑定 -- 写操作的参数必须与"最早从用户输入解析出的需求"一致，
   防止上下文里被污染的文本（间接提示注入）改变供应商、金额、付款账号等。

策略层是唯一的放行判断点：工具函数本身不判断权限，沙箱只能被策略放行后的调用进入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Protocol


class ToolKind(str, Enum):
    READ = "read"
    WRITE = "write"


class Risk(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


RISK_ORDER: dict[Risk, int] = {
    Risk.NONE: 0,
    Risk.LOW: 1,
    Risk.MEDIUM: 2,
    Risk.HIGH: 3,
}

# 审批等级：1 = 经办确认；2 = 主管确认；3 = 高管 / 招标流程
APPROVAL_LABEL: dict[int, str] = {
    1: "经办人确认",
    2: "主管确认（高风险）",
    3: "高管审批 + 招标流程",
}


# ---------------------------------------------------------------------------
# 审批人权限矩阵
# ---------------------------------------------------------------------------
# 角色 -> 该角色最高能签批的等级。审批等级定义见 APPROVAL_LABEL。
ROLE_AUTHORITY: dict[str, int] = {
    "buyer": 1,                # 采购经办
    "category_manager": 2,     # 品类主管
    "procurement_manager": 2,  # 采购经理
    "finance_manager": 2,      # 财务经理
    "vp": 3,                   # 分管副总
    "general_manager": 3,      # 总经理
}

ROLE_LABELS: dict[str, str] = {
    "buyer": "采购经办",
    "category_manager": "品类主管",
    "procurement_manager": "采购经理",
    "finance_manager": "财务经理",
    "vp": "分管副总",
    "general_manager": "总经理",
}


def role_label(role: str) -> str:
    return f"{ROLE_LABELS.get(role, role)}（{role}）"


def roles_for_level(level: int, exclude: Iterable[str] | None = None) -> list[str]:
    """列出有权签批该等级的角色（可排除已签过的角色，用于会签）。"""
    blocked = set(exclude or ())
    return [role for role, cap in ROLE_AUTHORITY.items() if cap >= level and role not in blocked]


def required_approvals(level: int) -> int:
    """需要几名审批人。3 级（≥100 万或触发招标）要求两人会签。"""
    return 2 if level >= 3 else 1


def role_can_sign(role: str, level: int) -> bool:
    return ROLE_AUTHORITY.get(role, 0) >= level


# 受保护的供应商主数据字段：银行账号等只能走财务主数据流程，Agent 一律不得修改
PROTECTED_SUPPLIER_FIELDS = {
    "bank_account",
    "bank_name",
    "payment_account",
    "invoice_account",
    "tax_id",
    "legal_person",
}


class StoreView(Protocol):
    def supplier(self, supplier_id: str) -> dict[str, Any] | None: ...


@dataclass
class WriteContext:
    """写操作放行所需的上下文，只包含"可信来源"的信息。"""

    store: StoreView
    requirement: dict[str, Any] | None = None
    category: str | None = None
    candidate_supplier_ids: set[str] = field(default_factory=set)
    retrieved: set[tuple[str, str]] = field(default_factory=set)
    injection_events: int = 0


@dataclass
class PolicyDecision:
    allowed: bool
    needs_approval: bool
    risk: Risk
    approval_level: int
    code: str
    reason: str
    warnings: list[str] = field(default_factory=list)
    binding: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "needs_approval": self.needs_approval,
            "risk": self.risk.value,
            "approval_level": self.approval_level,
            "code": self.code,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "binding": dict(self.binding),
        }


def base_approval_level(risk: Risk) -> int:
    return {Risk.NONE: 0, Risk.LOW: 1, Risk.MEDIUM: 1, Risk.HIGH: 2}[risk]


def confirmation_token(tool_name: str, approval_level: int) -> str | None:
    """高风险操作要求输入确认口令，避免误触或顺手回车。"""
    if approval_level < 2:
        return None
    return f"APPROVE-{tool_name.upper().replace('_', '-')}"


def authorize_read(tool_name: str) -> PolicyDecision:
    """只读操作自动执行。"""
    return PolicyDecision(
        allowed=True,
        needs_approval=False,
        risk=Risk.NONE,
        approval_level=0,
        code="read_auto_allowed",
        reason="只读操作，按策略自动执行",
    )


def _deny(risk: Risk, code: str, reason: str, **kw: Any) -> PolicyDecision:
    return PolicyDecision(
        allowed=False,
        needs_approval=False,
        risk=risk,
        approval_level=0,
        code=code,
        reason=reason,
        **kw,
    )


def authorize_write(
    tool_name: str, risk: Risk, args: dict[str, Any], ctx: WriteContext
) -> PolicyDecision:
    """写操作的唯一放行入口。任何一条不满足 -> 直接拒绝，不给"先执行后补"的空间。"""
    warnings: list[str] = []

    # ---- 0. 参数基本校验 -------------------------------------------------
    amount = args.get("amount")
    if amount is not None and (not isinstance(amount, (int, float)) or amount <= 0):
        return _deny(risk, "invalid_amount", f"金额参数不合法：{amount!r}")

    # ---- 1. 来源绑定：写操作必须与最初解析出的需求一致 ----------------------
    req = ctx.requirement
    if req is None:
        return _deny(risk, "missing_requirement", "没有可信需求上下文，拒绝执行写操作")

    binding: dict[str, Any] = {}

    requested_category = args.get("category")
    if requested_category and ctx.category and requested_category != ctx.category:
        return _deny(
            risk,
            "origin_mismatch_category",
            f"写操作品类（{requested_category}）与最初识别的品类（{ctx.category}）不一致",
        )
    if ctx.category:
        binding["category"] = ctx.category

    supplier_id = args.get("supplier_id")
    if supplier_id:
        supplier = ctx.store.supplier(str(supplier_id))
        if supplier is None:
            return _deny(risk, "unknown_supplier", f"供应商 {supplier_id} 不存在")
        status = str(supplier.get("avl_status", "unknown"))
        if status == "blocked":
            return _deny(risk, "blocked_supplier", f"供应商 {supplier_id} 已被冻结，禁止交易")
        if status == "not_listed" and not req.get("single_source"):
            return _deny(
                risk,
                "supplier_not_in_avl",
                f"供应商 {supplier_id} 不在合格供应商名录内，且未提交单一来源特批",
            )
        if status == "not_listed" and req.get("single_source"):
            warnings.append(f"供应商 {supplier_id} 为名录外，需按单一来源特批流程处理")
        # 允许集合 = 本次检索出的候选 ∪ 用户在原始输入里点名的供应商。
        # 这样既挡住"模型被注入后自己编一个供应商"，又不误伤合法的指定/单一来源采购。
        preferred = str(req.get("preferred_supplier_id") or "").upper()
        allowlist = set(ctx.candidate_supplier_ids)
        if preferred:
            allowlist.add(preferred)
        if allowlist and str(supplier_id) not in allowlist:
            return _deny(
                risk,
                "supplier_not_candidate",
                f"供应商 {supplier_id} 既不在本次检索出的候选名单内，也不是用户点名的供应商，"
                "疑似被上下文篡改",
            )
        binding["supplier_id"] = str(supplier_id)
        binding["supplier_avl_status"] = status

    budget = req.get("budget_amount")
    if isinstance(amount, (int, float)) and isinstance(budget, (int, float)) and budget > 0:
        if amount > budget * 1.05:
            return _deny(
                risk,
                "origin_mismatch_amount",
                f"写操作金额 {amount:,.0f} 超出最初需求预算 {budget:,.0f} 的 5% 以上",
            )
        binding["budget_amount"] = budget

    if args.get("bank_account"):
        return _deny(
            risk,
            "forbidden_field",
            "写操作不允许携带银行账号字段，付款信息必须走财务系统主数据流程",
        )

    target_field = str(args.get("field", "") or "")
    if target_field and (
        target_field in PROTECTED_SUPPLIER_FIELDS
        or "银行" in target_field
        or "账号" in target_field
    ):
        return _deny(
            risk,
            "protected_field",
            f"字段 {target_field} 属于受保护主数据，禁止由 Agent 修改，须走财务主数据流程",
        )

    # ---- 2. 风险分级与审批等级 -------------------------------------------
    approval_level = base_approval_level(risk)
    if isinstance(amount, (int, float)) and amount >= 1_000_000:
        approval_level = max(approval_level, 3)
        warnings.append("金额达到 100 万元以上，按 R-PRC-003 需公开招标并由分管副总审批")

    if ctx.injection_events > 0 and RISK_ORDER[risk] >= RISK_ORDER[Risk.HIGH]:
        approval_level = max(approval_level, 3)
        warnings.append("本次会话检测到疑似提示注入，高风险操作已提升审批等级")

    return PolicyDecision(
        allowed=True,
        needs_approval=True,
        risk=risk,
        approval_level=approval_level,
        code="approval_required",
        reason=f"写操作默认人工审批：{APPROVAL_LABEL[approval_level]}",
        warnings=warnings,
        binding=binding,
    )