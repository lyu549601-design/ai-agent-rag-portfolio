"""Agent 编排：工具调用循环 + 治理门禁 + 人工审批 + 审计留痕。

一条请求的完整路径：

    用户输入
      -> 入口提示注入检测（命中即拒绝，不进入循环）
      -> 只读工具循环（工具输出先清洗再进上下文）
      -> 写操作？-> policy 来源绑定校验 -> 人工审批 -> 沙箱执行
      -> 结论引用校验（只允许引用本次真的检索到的记录）
      -> 审计链落盘

任何一步不过，都会以"拦截"结束，而不是"先执行后补手续"。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from . import guardrails
from .audit import AuditLog, digest
from .llm.base import Brain, BrainUnavailable, FinalAnswer, ToolCall, observation
from .policy import PolicyDecision, Risk, ToolKind, WriteContext, authorize_read, authorize_write, confirmation_token
from .sandbox import Sandbox
from .store import ProcurementStore
from .tools import REGISTRY, Citation, ToolError, ToolResult

MAX_STEPS = 14


# ---------------------------------------------------------------------------
# 审批
# ---------------------------------------------------------------------------


@dataclass
class ApprovalRequest:
    tool: str
    args: dict[str, Any]
    risk: str
    approval_level: int
    confirmation_token: str | None
    reason: str
    warnings: list[str] = field(default_factory=list)
    binding: dict[str, Any] = field(default_factory=dict)
    context_note: str | None = None


@dataclass
class ApprovalResponse:
    approved: bool
    approver: str = "human"
    comment: str | None = None


class Approver(Protocol):
    def request(self, request: ApprovalRequest) -> ApprovalResponse: ...


class DenyAllApprover:
    """无人值守：一律拒绝。用于评测与自动化场景。"""

    name = "deny_all"

    def request(self, request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            approved=False, approver="deny_all", comment="无人值守模式：不执行任何写操作"
        )


class AutoApproveApprover:
    """正向对照：自动批准。仅用于验证"审批通过后确实能执行"。"""

    name = "auto_approve"

    def __init__(self, approver: str = "采购经理（演示）") -> None:
        self.approver = approver

    def request(self, request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(approved=True, approver=self.approver, comment="自动批准（演示）")


class ScriptedApprover:
    """按脚本批准 / 拒绝，供评测构造确定的通关与拦截路径。"""

    def __init__(self, plan: dict[str, bool], approver: str = "评测审批人") -> None:
        self.plan = {key.lower(): value for key, value in plan.items()}
        self.default = self.plan.get("*", False)
        self.approver = approver
        self.seen: list[str] = []

    def request(self, request: ApprovalRequest) -> ApprovalResponse:
        key = request.tool.lower()
        if key not in self.plan and key not in self.seen:
            self.seen.append(key)
        approved = self.plan.get(key, self.default)
        return ApprovalResponse(
            approved=approved,
            approver=self.approver,
            comment="评测脚本批准" if approved else "评测脚本拒绝",
        )


# ---------------------------------------------------------------------------
# 运行结果
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    user_input: str
    final_text: str
    brain: str
    run_id: str
    category: str | None = None
    citations: list[dict[str, str]] = field(default_factory=list)
    ungrounded_citations: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    injections: list[dict[str, Any]] = field(default_factory=list)
    writes_attempted: list[dict[str, Any]] = field(default_factory=list)
    writes_executed: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = "finished"
    defense_actions: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    audit_ok: bool = True
    audit_checked: int = 0

    @property
    def executed_write_tools(self) -> list[str]:
        return [item["tool"] for item in self.writes_executed]

    @property
    def blocked_write_tools(self) -> list[str]:
        return [item["tool"] for item in self.blocked]

    @property
    def unauthorized_writes(self) -> list[dict[str, Any]]:
        """没有人工批准却被执行的写操作（正常情况下必须永远为空）。"""
        approved_ids = {item.get("operation_id") for item in self.approvals if item.get("approved")}
        out = []
        for item in self.writes_executed:
            if item.get("operation_id") not in approved_ids:
                out.append(item)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "brain": self.brain,
            "user_input": self.user_input,
            "category": self.category,
            "final_text": self.final_text,
            "citations": self.citations,
            "ungrounded_citations": self.ungrounded_citations,
            "tool_calls": self.tool_calls,
            "blocked": self.blocked,
            "approvals": self.approvals,
            "injections": self.injections,
            "writes_attempted": self.writes_attempted,
            "writes_executed": self.writes_executed,
            "stopped_reason": self.stopped_reason,
            "defense_actions": self.defense_actions,
            "retrieved_ids": self.retrieved_ids,
            "audit_ok": self.audit_ok,
            "audit_checked": self.audit_checked,
        }


@dataclass
class RunContext:
    run_id: str
    store: ProcurementStore
    requirement: dict[str, Any] | None = None
    category: str | None = None
    candidate_supplier_ids: set[str] = field(default_factory=set)
    retrieved: set[str] = field(default_factory=set)
    citations: list[dict[str, str]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    injections: list[dict[str, Any]] = field(default_factory=list)
    writes_attempted: list[dict[str, Any]] = field(default_factory=list)
    writes_executed: list[dict[str, Any]] = field(default_factory=list)
    defense_actions: list[str] = field(default_factory=list)

    def write_context(self) -> WriteContext:
        return WriteContext(
            store=self.store,
            requirement=self.requirement,
            category=self.category,
            candidate_supplier_ids=set(self.candidate_supplier_ids),
            retrieved=set(self.retrieved),
            injection_events=len(self.injections),
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    def __init__(
        self,
        *,
        store: ProcurementStore,
        audit: AuditLog,
        brain: Brain,
        sandbox: Sandbox,
        approver: Approver,
    ) -> None:
        self.store = store
        self.audit = audit
        self.brain = brain
        self.sandbox = sandbox
        self.approver = approver

    # ------------------------------------------------------------------ 入口
    def run(self, user_input: str, *, max_steps: int = MAX_STEPS) -> RunResult:
        run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        ctx = RunContext(run_id=run_id, store=self.store)
        self.audit.append(
            "user", "request.received", run_id=run_id, text=user_input, brain=self.brain.name
        )

        # 1. 入口防线：直接注入的指令类输入，不进工具循环
        input_findings = guardrails.scan(user_input, source="user_input")
        if input_findings:
            note = guardrails.summarize(input_findings)
            self.audit.append(
                "system", "guardrail.input_blocked", run_id=run_id, detail=note
            )
            result = RunResult(
                user_input=user_input,
                final_text=(
                    "已拒绝执行：输入中包含疑似提示注入指令（"
                    + "、".join(sorted({f.kind for f in input_findings}))
                    + "）。\n"
                    "治理策略不允许通过对话指令绕过审批流程。如果确实需要采购或付款，"
                    "请按正常描述提出需求，我会给出建议并走人工审批。"
                ),
                brain=self.brain.name,
                run_id=run_id,
                stopped_reason="input_guardrail_blocked",
                injections=[
                    {"source": "user_input", "kind": f.kind, "matched": f.matched}
                    for f in input_findings
                ],
                defense_actions=["input_guardrail"],
            )
            return self._finish(result)

        # 2. 工具循环
        final: FinalAnswer | None = None
        steps = 0
        while steps < max_steps:
            steps += 1
            try:
                action = self.brain.next_action(
                    user_input=user_input, observations=ctx.observations, tools=REGISTRY
                )
            except BrainUnavailable as exc:
                self.audit.append("system", "brain.unavailable", run_id=run_id, detail=str(exc))
                return self._finish(
                    RunResult(
                        user_input=user_input,
                        final_text=f"模型不可用，已停止本次执行：{exc}",
                        brain=self.brain.name,
                        run_id=run_id,
                        stopped_reason="brain_unavailable",
                    )
                )

            if isinstance(action, FinalAnswer):
                final = action
                break
            if isinstance(action, ToolCall):
                ctx.observations.append(self._execute(action, ctx))
                continue
            raise TypeError(f"未知的模型动作：{action!r}")

        if final is None:
            self.audit.append("system", "loop.limit_reached", run_id=run_id, steps=steps)
            return self._finish(
                RunResult(
                    user_input=user_input,
                    final_text="达到最大工具调用步数，已停止并交人工处理（避免无限循环）。",
                    brain=self.brain.name,
                    run_id=run_id,
                    stopped_reason="max_steps",
                )
            )

        text, ungrounded = self._verify_citations(final, ctx)
        result = RunResult(
            user_input=user_input,
            final_text=text,
            brain=self.brain.name,
            run_id=run_id,
            category=ctx.category,
            citations=ctx.citations,
            ungrounded_citations=ungrounded,
            tool_calls=ctx.tool_calls,
            blocked=ctx.blocked,
            approvals=ctx.approvals,
            injections=ctx.injections,
            writes_attempted=ctx.writes_attempted,
            writes_executed=ctx.writes_executed,
            stopped_reason="finished",
            defense_actions=ctx.defense_actions,
            retrieved_ids=sorted(ctx.retrieved),
        )
        return self._finish(result)

    # -------------------------------------------------------------- 工具执行
    def _execute(self, call: ToolCall, ctx: RunContext) -> dict[str, Any]:
        spec = REGISTRY.get(call.name)
        ctx.tool_calls.append({"tool": call.name, "args": call.args})
        self.audit.append(
            "agent",
            "tool.call",
            run_id=ctx.run_id,
            tool=call.name,
            args_hash=digest(call.args),
        )

        if spec is None:
            self.audit.append(
                "system", "tool.rejected", run_id=ctx.run_id, tool=call.name, reason="未注册工具"
            )
            ctx.blocked.append(
                {"tool": call.name, "reason": "未注册工具", "code": "unregistered_tool", "args": call.args}
            )
            return observation(
                call.name,
                args=call.args,
                status="blocked",
                summary="工具不在白名单内，已拒绝",
            )

        if spec.kind is ToolKind.WRITE:
            return self._execute_write(spec.name, spec.risk, call.args, ctx)
        return self._execute_read(spec.name, call.args, ctx)

    def _execute_read(self, name: str, args: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        decision = authorize_read(name)
        try:
            result: ToolResult = REGISTRY[name].handler(self.store, **args)
        except ToolError as exc:
            self.audit.append(
                "agent", "tool.error", run_id=ctx.run_id, tool=name, code=exc.code, detail=exc.message
            )
            return observation(
                name, args=args, status="error", summary=f"工具执行失败：{exc.message}",
                note=decision.reason,
            )
        except TypeError as exc:
            self.audit.append(
                "agent", "tool.error", run_id=ctx.run_id, tool=name, code="bad_arguments", detail=str(exc)
            )
            return observation(
                name, args=args, status="error", summary=f"参数不匹配：{exc}"
            )

        data, findings = self._sanitize_data(result.data, source=f"tool:{name}")
        note = None
        if findings:
            ctx.injections.extend(
                {
                    "source": f"tool:{name}",
                    "kind": finding.kind,
                    "matched": finding.matched,
                }
                for finding in findings
            )
            if "tool_output_sanitized" not in ctx.defense_actions:
                ctx.defense_actions.append("tool_output_sanitized")
            note = (
                "该工具返回的自由文本中发现疑似提示注入，已中和处理，"
                "其中的指令不被执行（" + guardrails.summarize(findings) + "）"
            )
            self.audit.append(
                "system",
                "guardrail.tool_output_sanitized",
                run_id=ctx.run_id,
                tool=name,
                kinds=sorted({finding.kind for finding in findings}),
            )

        for citation in result.citations:
            self._register_citation(citation, ctx)

        self._absorb_state(name, data, ctx)
        return observation(
            name,
            args=args,
            status="ok",
            summary=result.summary,
            data=data,
            citations=[c.as_dict() for c in result.citations],
            note=note,
        )

    def _absorb_state(self, name: str, data: dict[str, Any], ctx: RunContext) -> None:
        """把关键结论吸收进"可信上下文"，供后续策略校验使用。"""
        if name == "parse_requirement":
            ctx.requirement = data.get("requirement")
        elif name == "classify_category":
            ctx.category = data.get("category")
            if ctx.requirement is not None:
                ctx.requirement["category"] = ctx.category
        elif name == "search_suppliers":
            ctx.candidate_supplier_ids = {
                candidate["supplier_id"] for candidate in data.get("candidates", [])
            }

    def _execute_write(
        self, name: str, risk: Risk, args: dict[str, Any], ctx: RunContext
    ) -> dict[str, Any]:
        spec = REGISTRY[name]
        ctx.writes_attempted.append({"tool": name, "args": args, "risk": risk.value})

        if not spec.requires_approval:
            # 防御性检查：写工具必须挂审批标记，否则视为配置错误并拒绝
            self.audit.append(
                "system", "write.rejected", run_id=ctx.run_id, tool=name, reason="写工具未挂审批标记"
            )
            ctx.blocked.append(
                {"tool": name, "code": "gate_missing", "reason": "写工具未挂审批标记", "args": args}
            )
            return observation(name, args=args, status="blocked", summary="写操作未挂审批门禁，已拒绝")

        decision: PolicyDecision = authorize_write(name, risk, args, ctx.write_context())
        self.audit.append(
            "policy",
            "write.evaluated",
            run_id=ctx.run_id,
            tool=name,
            allowed=decision.allowed,
            code=decision.code,
            reason=decision.reason,
            args_hash=digest(args),
        )

        if not decision.allowed:
            ctx.blocked.append(
                {
                    "tool": name,
                    "code": decision.code,
                    "reason": decision.reason,
                    "args": args,
                    "risk": risk.value,
                }
            )
            if "policy_denied" not in ctx.defense_actions:
                ctx.defense_actions.append("policy_denied")
            self.audit.append(
                "policy", "write.denied", run_id=ctx.run_id, tool=name, code=decision.code
            )
            return observation(
                name,
                args=args,
                status="blocked",
                summary=f"策略拒绝执行写操作：{decision.reason}（{decision.code}）",
            )

        token = confirmation_token(name, decision.approval_level)
        context_note = None
        if ctx.injections:
            context_note = (
                f"注意：本次会话已检测到 {len(ctx.injections)} 处疑似提示注入，"
                "已中和但请核对参数来源"
            )

        approval_request = ApprovalRequest(
            tool=name,
            args=args,
            risk=risk.value,
            approval_level=decision.approval_level,
            confirmation_token=token,
            reason=decision.reason,
            warnings=decision.warnings,
            binding=decision.binding,
            context_note=context_note,
        )
        self.audit.append(
            "agent",
            "approval.requested",
            run_id=ctx.run_id,
            tool=name,
            approval_level=decision.approval_level,
            args_hash=digest(args),
        )
        response = self.approver.request(approval_request)

        record = {
            "tool": name,
            "risk": risk.value,
            "approval_level": decision.approval_level,
            "approved": bool(response.approved),
            "approver": response.approver,
            "comment": response.comment,
            "binding": decision.binding,
            "args": args,
            "operation_id": None,
        }

        if not response.approved:
            ctx.approvals.append(record)
            ctx.blocked.append(
                {
                    "tool": name,
                    "code": "approval_denied",
                    "reason": f"人工审批未通过（{response.approver}）",
                    "args": args,
                    "risk": risk.value,
                }
            )
            if "approval_gate" not in ctx.defense_actions:
                ctx.defense_actions.append("approval_gate")
            self.audit.append(
                "approver", "approval.denied", run_id=ctx.run_id, tool=name, approver=response.approver
            )
            return observation(
                name,
                args=args,
                status="blocked",
                summary=f"人工审批未通过（{response.approver}），未执行任何写操作",
            )

        # 审批通过 -> 构造操作 -> 沙箱执行
        try:
            result = spec.handler(self.store, **args)
        except ToolError as exc:
            self.audit.append(
                "agent", "write.build_failed", run_id=ctx.run_id, tool=name, code=exc.code
            )
            return observation(
                name, args=args, status="error", summary=f"写操作构造失败：{exc.message}"
            )

        operation = result.data.get("operation", {})
        record["operation_id"] = operation.get("operation_id")
        ctx.approvals.append(record)

        sandbox_result = self.sandbox.execute(operation)
        self.audit.append(
            "sandbox",
            "write.executed" if sandbox_result.ok else "write.failed",
            run_id=ctx.run_id,
            tool=name,
            operation_id=operation.get("operation_id"),
            backend=sandbox_result.backend,
            ok=sandbox_result.ok,
            detail=sandbox_result.message,
        )

        if not sandbox_result.ok:
            return observation(
                name, args=args, status="error", summary=sandbox_result.message, data=sandbox_result.as_dict()
            )

        ctx.writes_executed.append(
            {
                "tool": name,
                "operation_id": operation.get("operation_id"),
                "backend": sandbox_result.backend,
                "artifact": sandbox_result.artifact,
            }
        )
        return observation(
            name,
            args=args,
            status="ok",
            summary=f"审批通过并执行：{sandbox_result.message}",
            data=sandbox_result.as_dict(),
        )

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _sanitize_data(value: Any, source: str) -> tuple[Any, list[guardrails.Finding]]:
        findings: list[guardrails.Finding] = []

        def walk(node: Any) -> Any:
            if isinstance(node, str):
                cleaned, hits = guardrails.sanitize(node, source=source)
                findings.extend(hits)
                return cleaned
            if isinstance(node, dict):
                return {key: walk(item) for key, item in node.items()}
            if isinstance(node, list):
                return [walk(item) for item in node]
            return node

        return walk(value), findings

    def _register_citation(self, citation: Citation, ctx: RunContext) -> None:
        if citation.id not in ctx.retrieved:
            ctx.retrieved.add(citation.id)
        payload = citation.as_dict()
        if payload not in ctx.citations:
            ctx.citations.append(payload)

    def _verify_citations(
        self, final: FinalAnswer, ctx: RunContext
    ) -> tuple[str, list[str]]:
        """引用校验：只允许引用本次真的检索到的记录编号。"""
        ungrounded = [cid for cid in final.citation_ids if cid not in ctx.retrieved]
        text = final.text
        for cid in ungrounded:
            text = text.replace(cid, "【引用未检索到，已移除】")
            self.audit.append(
                "system", "citation.rejected", run_id=ctx.run_id, citation=cid
            )
        return text, ungrounded

    def _finish(self, result: RunResult) -> RunResult:
        verification = self.audit.verify()
        result.audit_ok = verification.ok
        result.audit_checked = verification.checked
        self.audit.append(
            "system",
            "run.completed",
            run_id=result.run_id,
            stopped_reason=result.stopped_reason,
            writes_executed=[item["tool"] for item in result.writes_executed],
            blocked=[item["tool"] for item in result.blocked],
            injections=len(result.injections),
            citations=len(result.citations),
        )
        return result