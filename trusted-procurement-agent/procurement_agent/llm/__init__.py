"""大脑工厂：按配置返回 DeepSeek 或离线模拟大脑。"""

from __future__ import annotations

from .base import Action, Brain, BrainUnavailable, FinalAnswer, ToolCall, observation
from .offline import OfflineBrain

__all__ = [
    "Action",
    "Brain",
    "BrainUnavailable",
    "FinalAnswer",
    "ToolCall",
    "OfflineBrain",
    "observation",
    "build_brain",
]


def build_brain(name: str = "auto", *, allow_fallback: bool = True) -> tuple[Brain, str]:
    """返回 (brain, 说明)。name 取 auto / offline / deepseek。"""
    normalized = (name or "auto").strip().lower()

    if normalized == "offline":
        return OfflineBrain(), "使用离线模拟大脑（规则驱动，可复现，不需要密钥）"

    if normalized == "deepseek":
        from .deepseek import DeepSeekBrain

        try:
            return DeepSeekBrain(), "使用 DeepSeek 大脑"
        except BrainUnavailable as exc:
            if not allow_fallback:
                raise
            return OfflineBrain(), f"DeepSeek 不可用，已回退到离线大脑：{exc}"

    # auto
    import os

    if os.environ.get("DEEPSEEK_API_KEY"):
        from .deepseek import DeepSeekBrain

        try:
            return DeepSeekBrain(), "检测到 DEEPSEEK_API_KEY，使用 DeepSeek 大脑"
        except BrainUnavailable as exc:
            return OfflineBrain(), f"DeepSeek 初始化失败，已回退到离线大脑：{exc}"

    return OfflineBrain(), "未配置密钥，使用离线模拟大脑"