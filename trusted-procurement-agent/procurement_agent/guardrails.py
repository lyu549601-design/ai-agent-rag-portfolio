"""提示注入（prompt injection）防护。

两处防线：

1. 入口防线 -- 用户输入先扫描，命中即拒绝执行，不进入工具循环；
2. 数据防线 -- 工具返回的内容（供应商备注、订单备注、合同附件文字等）
   一律视为不可信，注入模型上下文前先扫描并中和。

命中不改变权限：即使攻击者成功把文字塞进上下文，写操作依然要过审批与来源校验。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 规则库：正则 + 攻击类型。命中即判定为疑似注入。
PATTERNS: list[tuple[str, str]] = [
    (r"忽略(以上|之前|上面|前面|所有)[^\n]{0,8}(规则|指令|提示|要求|限制)", "ignore_instructions"),
    (r"无视[^\n]{0,8}(规则|指令|限制|要求)", "ignore_instructions"),
    (r"(跳过|绕过|不用|不需要|无需|免去)[^\n]{0,6}(审批|审核|确认|人工|复核)", "bypass_approval"),
    (r"直接(下单|付款|打款|审批|创建|执行|发货|采购)", "direct_write"),
    (r"你现在(是|变成|扮演|作为)", "role_hijack"),
    (r"(开发者|开发人员|上帝|超级|root|DAN)[\s]*模式", "role_hijack"),
    (r"(system\s*prompt|系统提示词|系统提示|系统指令|初始指令)", "prompt_leak"),
    (r"覆盖[^\n]{0,8}(规则|策略|配置|权限)", "override_policy"),
    (r"把[^\n]{0,10}(付款|收款|银行|打款)[^\n]{0,6}(账号|账户|信息)?[^\n]{0,4}改", "payment_redirect"),
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?)", "ignore_instructions"),
    (r"disregard\s+[^\n]{0,24}(rules?|polic(y|ies)|instructions?)", "ignore_instructions"),
    (r"bypass\s+(the\s+)?(approval|review|human|confirm\w*)", "bypass_approval"),
    (r"without\s+(any\s+)?(approval|human\s+confirmation)", "bypass_approval"),
    (r"you\s+are\s+now\b", "role_hijack"),
    (r"\bnew\s+instructions?\b", "override_policy"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), kind) for p, kind in PATTERNS]

REDACTION = "[[已拦截：疑似提示注入内容]]"


@dataclass(frozen=True)
class Finding:
    kind: str
    matched: str
    source: str = "unknown"


def scan(text: str, source: str = "unknown") -> list[Finding]:
    """扫描文本，返回所有命中项。"""
    if not text:
        return []
    findings: list[Finding] = []
    seen: set[tuple[str, int]] = set()
    for pattern, kind in _COMPILED:
        for match in pattern.finditer(text):
            key = (kind, match.start())
            if key in seen:
                continue
            seen.add(key)
            findings.append(Finding(kind=kind, matched=match.group(0), source=source))
    return findings


def is_suspicious(text: str) -> bool:
    return bool(scan(text))


def sanitize(text: str, source: str = "unknown") -> tuple[str, list[Finding]]:
    """把命中片段替换成中性标记，返回 (中和后的文本, 命中列表)。"""
    if not text:
        return text, []
    findings = scan(text, source=source)
    if not findings:
        return text, []

    cleaned = text
    for pattern, _kind in _COMPILED:
        cleaned = pattern.sub(REDACTION, cleaned)
    return cleaned, findings


def strip_untrusted(text: str, source: str = "unknown") -> tuple[str, list[Finding]]:
    """工具输出专用入口：返回中和后的文本与命中记录。"""
    return sanitize(text, source=source)


def summarize(findings: list[Finding]) -> str:
    if not findings:
        return "未发现疑似提示注入"
    kinds = sorted({f.kind for f in findings})
    sample = findings[0].matched
    return f"命中 {len(findings)} 处疑似提示注入（类型：{', '.join(kinds)}；样例：{sample!r}）"