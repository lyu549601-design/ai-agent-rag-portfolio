"""可插拔模型层。

Agent 与模型之间只通过 ``Brain`` 接口耦合，两种实现共用同一条工具调用循环：

* ``OfflineBrain``  -- 离线模拟大脑，规则驱动的确定性计划，不需要任何密钥；
* ``DeepSeekBrain`` -- 真实 DeepSeek 模型，走 OpenAI 兼容的 function calling。

无论用哪个大脑，写操作都要过 policy + 人工审批 + 沙箱，模型无法绕过。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """请求调用一个工具。

    ``meta`` 用来携带"必须原样回传给模型"的上下文（例如思考模式模型返回的
    ``reasoning_content``）。离线大脑不使用它。
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalAnswer:
    """给出最终回答。citation_ids 会被 Agent 校验是否真的检索过。"""

    text: str
    citation_ids: list[str] = field(default_factory=list)


Action = ToolCall | FinalAnswer


class BrainUnavailable(RuntimeError):
    """模型不可用（缺密钥、网络异常、返回不可解析等）。"""


class Brain(Protocol):
    name: str

    def next_action(
        self,
        *,
        user_input: str,
        observations: list[dict[str, Any]],
        tools: dict[str, Any],
    ) -> Action: ...


def observation(
    tool: str,
    args: dict[str, Any] | None = None,
    status: str = "ok",
    summary: str = "",
    data: dict[str, Any] | None = None,
    citations: list[dict[str, str]] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """构造一条工具观察结果，交给大脑决定下一步。"""
    record: dict[str, Any] = {
        "tool": tool,
        "args": args or {},
        "status": status,
        "summary": summary,
        "data": data or {},
        "citations": citations or [],
    }
    if note:
        record["note"] = note
    return record