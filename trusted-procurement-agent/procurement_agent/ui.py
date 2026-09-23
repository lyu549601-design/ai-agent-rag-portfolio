"""终端输出工具：颜色、排版、统一提示语。

不依赖任何第三方库；在管道 / 非交互环境下自动关闭颜色。
"""

from __future__ import annotations

import os
import sys

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}

_ENABLED: bool | None = None


def _enable_windows_ansi() -> bool:
    """在 Windows 控制台打开 ANSI 转义支持。"""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        return True
    except Exception:
        return False


def color_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = bool(
            sys.stdout.isatty()
            and not os.environ.get("NO_COLOR")
            and _enable_windows_ansi()
        )
    return _ENABLED


def set_color_enabled(value: bool | None) -> None:
    """显式开关颜色（None 表示恢复自动判断）。"""
    global _ENABLED
    _ENABLED = None if value is None else bool(value)


def c(text: object, *styles: str) -> str:
    """给文本加颜色 / 样式。颜色不可用时原样返回。"""
    text = str(text)
    if not styles or not color_enabled():
        return text
    prefix = "".join(_CODES.get(s, "") for s in styles)
    return f"{prefix}{text}{_CODES['reset']}"


def banner(text: str, width: int = 74) -> None:
    print()
    print(c("=" * width, "dim"))
    print(c(f"  {text}", "bold", "cyan"))
    print(c("=" * width, "dim"))


def section(text: str, width: int = 74) -> None:
    print()
    print(c(text, "bold", "magenta"))
    print(c("-" * width, "dim"))


def step(text: str) -> None:
    print(c("  -> ", "blue") + text)


def ok(text: str) -> None:
    print(c("  [OK] ", "green") + text)


def warn(text: str) -> None:
    print(c("  [!!] ", "yellow") + text)


def bad(text: str) -> None:
    print(c("  [XX] ", "red") + text)


def info(text: str) -> None:
    for line in str(text).splitlines() or [""]:
        print("       " + line)


def kv(key: str, value: object) -> None:
    print(c(f"       {key:<14}", "dim") + str(value))