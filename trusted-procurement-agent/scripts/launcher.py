#!/usr/bin/env python
"""一键启动 / 关闭网页演示。

通常不需要直接运行它 —— 双击项目根目录的「一键演示.bat」即可，
那个 bat 只是在调用本脚本。

为什么用脚本而不是纯 bat：启停需要做几件事，纯批处理写起来既脆又难维护：

1. 判断服务是不是**我们自己的**（而不是端口被别的程序占用）；
2. 启动后轮询健康检查，确认真的起来了再开浏览器；
3. 关闭时通过 HTTP 优雅关闭，而不是硬杀进程（硬杀杀错进程会更糟）；
4. 记 PID、写日志，出问题时有据可查。

对外命令：

    python scripts/launcher.py            # 开关式：没运行就启动，正在运行就问你要不要关
    python scripts/launcher.py --start
    python scripts/launcher.py --stop
    python scripts/launcher.py --status
    python scripts/launcher.py --port 8899
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace"
PID_FILE = WORKSPACE / "web.pid"
LOG_FILE = WORKSPACE / "web.log"

DEFAULT_PORT = 8765
SERVICE_NAME = "trusted-procurement-agent"
BAR = "=" * 62


def banner(title: str) -> None:
    print(BAR)
    print("  " + title)
    print(BAR)


def info(text: str) -> None:
    print("  " + text)


def service_status(port: int, timeout: float = 1.5) -> dict | None:
    """返回服务信息；不是我们的服务时返回 {'foreign': True}；没服务时返回 None。"""
    url = f"http://127.0.0.1:{port}/api/info"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if payload.get("service") == SERVICE_NAME:
        return payload
    return {"foreign": True}


def ensure_workspace() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)


def start(port: int, open_browser: bool = True, timeout: float = 30.0) -> int:
    banner("可信采购 Agent · 演示服务")
    status = service_status(port)

    if status and status.get("foreign"):
        info(f"端口 {port} 被其它程序占用了，换一个端口再试，例如：")
        info(f"  python scripts/launcher.py --port {port + 1}")
        return 2

    if status:
        info("演示服务已经在运行了，直接打开页面。")
    else:
        ensure_workspace()
        if not (ROOT / ".env").exists():
            info("提示：没有找到 .env，将使用离线模拟大脑（不需要密钥，全链路照常跑通）。")
        info("正在启动…")

        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        log_handle = LOG_FILE.open("a", encoding="utf-8")
        log_handle.write(f"\n--- 启动于 {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_handle.flush()

        process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-m", "procurement_agent", "web", "--port", str(port)],
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        PID_FILE.write_text(str(process.pid), encoding="utf-8")

        deadline = time.time() + timeout
        status = None
        while time.time() < deadline:
            if process.poll() is not None:
                info("服务进程启动后就退出了。日志最后几行：")
                _tail_log(8)
                return 1
            status = service_status(port, timeout=1.0)
            if status and not status.get("foreign"):
                break
            time.sleep(0.4)
            status = None

        if not status:
            info("启动超时。日志最后几行：")
            _tail_log(12)
            return 1

    url = f"http://127.0.0.1:{port}/"
    info("启动成功。")
    if status:
        info(f"大脑：{status.get('brain')}    沙箱：{status.get('sandbox')}")
        dataset = status.get("dataset") or {}
        if dataset:
            info(
                "数据：规则 {rules} · 供应商 {suppliers} · 订单 {orders} · 合同模板 {templates}".format(
                    rules=dataset.get("rules", "-"),
                    suppliers=dataset.get("suppliers", "-"),
                    orders=dataset.get("orders", "-"),
                    templates=dataset.get("templates", "-"),
                )
            )
    info(f"页面地址：{url}")
    info("关闭方式：再次双击「一键演示.bat」，或在终端执行：")
    info("  python scripts/launcher.py --stop")
    info(f"运行日志：{LOG_FILE}")
    print()

    if open_browser:
        webbrowser.open(url)
    return 0


def stop(port: int) -> int:
    banner("可信采购 Agent · 关闭演示服务")
    status = service_status(port)

    if status is None:
        info("没有检测到正在运行的演示服务（可能已经关了）。")
        PID_FILE.unlink(missing_ok=True)
        return 0
    if status.get("foreign"):
        info(f"端口 {port} 上跑的不是本项目的服务，我不动它。")
        return 2

    info("正在通知服务优雅退出…")
    closed = False
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/shutdown", data=b"{}", method="POST"
        )
        urllib.request.urlopen(request, timeout=5).read()
        closed = True
    except Exception:
        closed = False

    if not closed:
        info("优雅关闭没有成功，尝试按记录的进程号结束…")
        _kill_by_pidfile()

    deadline = time.time() + 15
    while time.time() < deadline:
        if service_status(port, timeout=0.8) is None:
            PID_FILE.unlink(missing_ok=True)
            info("已关闭。")
            return 0
        time.sleep(0.4)

    info("服务似乎还在运行；可以再执行一次 --stop，或手动关闭对应进程。")
    return 1


def _kill_by_pidfile() -> None:
    if not PID_FILE.exists():
        return
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def _tail_log(lines: int) -> None:
    if not LOG_FILE.exists():
        info("（没有日志文件）")
        return
    content = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in content[-lines:]:
        info("  " + line)


def status_report(port: int) -> int:
    banner("可信采购 Agent · 服务状态")
    status = service_status(port)
    if status is None:
        info("未运行。双击「一键演示.bat」即可启动。")
        return 1
    if status.get("foreign"):
        info(f"端口 {port} 被其它程序占用（不是本项目服务）。")
        return 2
    info(f"运行中：http://127.0.0.1:{port}/")
    info(f"大脑：{status.get('brain')}    沙箱：{status.get('sandbox')}")
    info(f"审计记录：{status.get('audit_records')} 条    交付出：{status.get('ledger_entries')} 条")
    return 0


def toggle(port: int) -> int:
    status = service_status(port)
    if status and not status.get("foreign"):
        banner("可信采购 Agent · 演示服务正在运行")
        info(f"页面地址：http://127.0.0.1:{port}/")
        print()
        try:
            answer = input("  要关闭它吗？[Y/n]：").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            info("没有收到输入，保持运行。")
            return 0
        if answer in ("", "y", "yes", "是", "关闭"):
            return stop(port)
        info("好的，保持运行。")
        return 0
    return start(port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="可信采购 Agent 演示服务：一键启动 / 关闭")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"端口，默认 {DEFAULT_PORT}")
    parser.add_argument("--start", action="store_true", help="只启动")
    parser.add_argument("--stop", action="store_true", help="只关闭")
    parser.add_argument("--status", action="store_true", help="只看状态")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args(argv)

    if args.start and args.stop:
        print("--start 与 --stop 不能同时用。")
        return 2
    if args.start:
        return start(args.port, open_browser=not args.no_browser)
    if args.stop:
        return stop(args.port)
    if args.status:
        return status_report(args.port)
    return toggle(args.port)


if __name__ == "__main__":
    raise SystemExit(main())