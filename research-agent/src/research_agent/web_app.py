from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .agents import ExecutorAgent
from .config import get_settings
from .graph import ResearchAgent
from .llm import DeepSeekLLM, LLMError
from .rag import KnowledgeBase
from .storage import MemoryStore, StorageError, build_database

settings = get_settings()
database = build_database(settings)
knowledge_base = KnowledgeBase(database)
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="企业知识研究助手", version="0.2.0")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return job


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path(__file__).with_name("web_ui.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> dict[str, Any]:
    result: dict[str, Any] = {
        "storage": {"ok": False, "detail": ""},
        "redis": {"ok": False, "detail": ""},
        "knowledge": {"documents": 0, "chunks": 0},
        "llm": {"configured": settings.llm_enabled, "model": settings.deepseek_model},
        "embedding": settings.embedding_model,
    }
    try:
        database.ping()
        result["storage"] = {"ok": True, "detail": settings.storage_backend}
    except Exception as exc:
        result["storage"] = {"ok": False, "detail": str(exc)}
    try:
        memory = MemoryStore(enabled=True, db=database, settings=settings, embedder=knowledge_base.embedder)
        memory.ping()
        result["redis"] = {"ok": True, "detail": "连接正常"}
    except Exception as exc:
        result["redis"] = {"ok": False, "detail": str(exc)}
    try:
        result["knowledge"] = {
            "documents": knowledge_base.document_count(),
            "chunks": database.chunk_count(),
        }
    except Exception as exc:
        result["knowledge"] = {"documents": 0, "chunks": 0, "error": str(exc)}
    return result


@app.get("/api/tasks")
def list_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for path in settings.reports_dir.glob("*.json"):
        payload = _read_json(path)
        if not payload or not payload.get("task_id"):
            continue
        task_id = str(payload["task_id"])
        if task_id.startswith(("eval-", "debug-", "check-", "smoke-")):
            continue
        created_at = payload.get("created_at") or datetime.fromtimestamp(path.stat().st_mtime).isoformat(
            timespec="seconds"
        )
        tasks.append(
            {
                "task_id": task_id,
                "topic": payload.get("topic", ""),
                "completed": bool(payload.get("completed")),
                "review_score": payload.get("review_score", 0.0),
                "memory_enabled": bool(payload.get("memory_enabled")),
                "elapsed_seconds": payload.get("elapsed_seconds", 0.0),
                "created_at": created_at,
                "report_path": str(settings.reports_dir / f"{task_id}.md"),
            }
        )
    return sorted(tasks, key=lambda item: item["created_at"], reverse=True)


@app.get("/api/report/{task_id}")
def get_report(task_id: str) -> dict[str, Any]:
    report_path = settings.reports_dir / f"{task_id}.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="报告不存在")
    result_path = settings.reports_dir / f"{task_id}.json"
    return {
        "task_id": task_id,
        "report": report_path.read_text(encoding="utf-8"),
        "result": _read_json(result_path) or {},
    }


@app.post("/api/jobs")
def create_job(payload: dict[str, Any]) -> dict[str, Any]:
    topic = str(payload.get("topic", "")).strip()
    if not topic:
        raise HTTPException(status_code=400, detail="研究主题不能为空")
    task_id = str(payload.get("task_id") or f"ui-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}")
    job = {
        "job_id": uuid.uuid4().hex,
        "task_id": task_id,
        "topic": topic,
        "memory_enabled": bool(payload.get("memory_enabled", True)),
        "fast": bool(payload.get("fast", False)),
        "status": "queued",
        "stage": "queued",
        "message": "任务已进入队列",
        "progress": 0.0,
        "subtasks": [],
        "result": None,
        "error": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with JOBS_LOCK:
        JOBS[job["job_id"]] = job
    threading.Thread(target=_run_job, args=(job["job_id"],), daemon=True).start()
    return job


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    with JOBS_LOCK:
        values = list(JOBS.values())
    return sorted(values, key=lambda item: item["created_at"], reverse=True)[:20]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        return dict(job)


@app.post("/api/ingest")
def reindex() -> dict[str, Any]:
    try:
        database.init_schema()
        stats = knowledge_base.ingest_dir(settings.knowledge_base_dir)
        return {"ok": True, **stats}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/knowledge")
def knowledge() -> dict[str, Any]:
    docs = []
    for path in sorted(settings.knowledge_base_dir.glob("*")):
        if path.is_file():
            docs.append({"name": path.name, "size": path.stat().st_size, "path": str(path)})
    return {
        "documents": docs,
        "document_count": len(docs),
        "chunks": database.chunk_count() if database.ping() else 0,
    }


@app.post("/api/ask")
def ask(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload.get("task_id", "")).strip()
    question = str(payload.get("question", "")).strip()
    if not task_id or not question:
        raise HTTPException(status_code=400, detail="任务和问题不能为空")
    result_path = settings.reports_dir / f"{task_id}.json"
    task = _read_json(result_path)
    if not task:
        raise HTTPException(status_code=404, detail="找不到任务结果")
    evidence = knowledge_base.search(question, limit=settings.retrieval_top_k, mode="hybrid")
    if not evidence:
        return {"answer": "知识库中没有找到可引用证据。", "citations": []}
    memory = MemoryStore(enabled=bool(task.get("memory_enabled", True)), db=database, settings=settings, embedder=knowledge_base.embedder)
    memory_context = memory.context(task_id, question) if task.get("memory_enabled", True) else ""
    context = ExecutorAgent._format_evidence(evidence)
    system = "你是企业知识研究助手。只能依据给定证据回答，引用必须使用真实 chunk_id。"
    user = f"""原研究主题：{task.get('topic', '')}
已有报告摘要：{str(task.get('report', ''))[:2500]}
跨步骤记忆：
{memory_context or '无'}

追问：{question}

可用证据：
{context}

请用中文给出简洁答案，每个关键结论后使用 [chunk_id] 引用。证据不足时明确说明。
"""
    try:
        answer = DeepSeekLLM(settings).chat(system, user, temperature=0.1)
    except LLMError:
        answer = "模型暂不可用，以下是检索到的证据：\n\n" + "\n\n".join(
            f"- [{chunk.chunk_id}] {chunk.title}：{chunk.content[:300]}" for chunk in evidence
        )
    valid_ids = {chunk.chunk_id for chunk in evidence}
    cited = ExecutorAgent.extract_citations(answer, valid_ids)
    if not cited:
        answer += "\n\n引用：" + "、".join(f"[{chunk.chunk_id}]" for chunk in evidence[:2])
    return {"answer": answer, "citations": [chunk.model_dump() for chunk in evidence]}


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    results = settings.evaluation_dir / "results"
    return {
        "retrieval": _read_json(results / "retrieval_metrics.json"),
        "memory": _read_json(results / "memory_ablation.json"),
    }


def _run_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        topic = job["topic"]
        task_id = job["task_id"]
        memory_enabled = job["memory_enabled"]
        fast = job["fast"]

    def progress(event: dict[str, Any]) -> None:
        with JOBS_LOCK:
            current = JOBS.get(job_id)
            if not current:
                return
            current.update(
                {
                    "status": event.get("stage", "running"),
                    "stage": event.get("stage", "running"),
                    "message": event.get("message", ""),
                    "progress": event.get("progress", 0.0),
                }
            )
            if event.get("subtasks"):
                current["subtasks"] = event["subtasks"]

    try:
        agent = ResearchAgent(settings, db=database, kb=knowledge_base)
        result = agent.run(
            topic,
            memory_enabled=memory_enabled,
            fast=fast,
            task_id=task_id,
            progress_callback=progress,
        )
        payload = result.model_dump(mode="json")
        payload["created_at"] = datetime.now().isoformat(timespec="seconds")
        payload["history"] = []
        MemoryStore(
            enabled=memory_enabled,
            db=database,
            settings=settings,
            embedder=knowledge_base.embedder,
        ).save_task(task_id, payload)
        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "completed",
                    "stage": "completed",
                    "message": "研究完成",
                    "progress": 1.0,
                    "result": payload,
                }
            )
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "failed",
                    "stage": "failed",
                    "message": f"任务失败：{exc}",
                    "progress": 1.0,
                    "error": str(exc),
                }
            )