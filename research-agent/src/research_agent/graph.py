from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import ExecutorAgent, PlannerAgent, ReviewerAgent, topological_layers
from .config import Settings, get_settings
from .llm import DeepSeekLLM
from .models import ReviewResult, RunResult, Subtask
from .rag import KnowledgeBase
from .storage import Database, MemoryStore, build_database


class ResearchState(TypedDict, total=False):
    topic: str
    task_id: str
    fast: bool
    memory_enabled: bool
    subtasks: list[Subtask]
    report: str
    review: dict
    retry_ids: list[str]
    feedback_by_task: dict[str, str]
    review_cycles: int
    errors: list[str]


class ResearchAgent:
    """LangGraph 主流程：Planner → 分层并行 Executor → Reviewer → 报告。"""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: DeepSeekLLM | None = None,
        db: Database | None = None,
        kb: KnowledgeBase | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or DeepSeekLLM(self.settings)
        self.db = db or build_database(self.settings)
        self.kb = kb or KnowledgeBase(self.db)
        self.planner = PlannerAgent(self.llm)
        self.reviewer = ReviewerAgent(self.llm)

    def run(
        self,
        topic: str,
        *,
        memory_enabled: bool = True,
        fast: bool = False,
        task_id: str | None = None,
        preset_tasks: list[Subtask] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> RunResult:
        started = time.perf_counter()
        task_id = task_id or f"ra-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        def emit(stage: str, message: str, progress: float, **extra: Any) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback({"stage": stage, "message": message, "progress": progress, **extra})
            except Exception:
                pass

        memory = MemoryStore(
            enabled=memory_enabled,
            db=self.db,
            settings=self.settings,
            embedder=self.kb.embedder,
        )
        executor = ExecutorAgent(self.kb, self.llm, memory)

        def plan_node(state: ResearchState) -> ResearchState:
            if state.get("subtasks"):
                return {"review_cycles": 0, "errors": [], "report": "", "retry_ids": [], "feedback_by_task": {}}
            tasks = self.planner.plan(state["topic"], fast=state.get("fast", False))
            emit("planned", f"已拆解为 {len(tasks)} 个研究任务", 0.15, subtasks=[task.model_dump() for task in tasks])
            return {
                "subtasks": tasks,
                "review_cycles": 0,
                "errors": [],
                "report": "",
                "retry_ids": [],
                "feedback_by_task": {},
            }

        def execute_node(state: ResearchState) -> ResearchState:
            tasks = list(state.get("subtasks", []))
            retry_ids = set(state.get("retry_ids", []))
            feedback_by_task = dict(state.get("feedback_by_task", {}))
            layers = topological_layers(tasks)
            for layer_index, layer in enumerate(layers, start=1):
                ready = [
                    task
                    for task in layer
                    if task.status == "pending" and (not retry_ids or task.id in retry_ids)
                ]
                if not ready:
                    continue
                emit(
                    "executing",
                    f"正在执行第 {layer_index}/{len(layers)} 层，共 {len(ready)} 个任务",
                    0.15 + 0.65 * layer_index / max(1, len(layers)),
                    subtasks=[task.model_dump() for task in tasks],
                )
                memory_contexts = {
                    task.id: memory.context(state["task_id"], f'{state["topic"]} {task.objective}')
                    if state["memory_enabled"]
                    else ""
                    for task in ready
                }
                updated = self._execute_layer(
                    executor=executor,
                    topic=state["topic"],
                    task_id=state["task_id"],
                    memory_enabled=state["memory_enabled"],
                    tasks=tasks,
                    ready=ready,
                    feedback_by_task=feedback_by_task,
                    memory_contexts=memory_contexts,
                )
                by_id = {item.id: item for item in updated}
                tasks = [by_id.get(task.id, task) for task in tasks]
            errors = [task.error for task in tasks if task.error]
            return {"subtasks": tasks, "retry_ids": [], "errors": errors}

        def review_node(state: ResearchState) -> ResearchState:
            tasks = list(state.get("subtasks", []))
            report = self._build_report(topic, tasks)
            emit("reviewing", "正在检查完整性、引用真实性和跨分支覆盖", 0.82)
            review = self.reviewer.review(
                topic=topic,
                tasks=tasks,
                report=report,
                known_chunk_ids={chunk_id for task in tasks for chunk_id in task.citations},
            )
            cycles = int(state.get("review_cycles", 0))
            if not review.passed and cycles < self.settings.max_review_retries:
                retry_ids = review.failed_task_ids or ([tasks[-1].id] if tasks else [])
                feedback = "；".join(review.reasons) or "请补充证据、加强引用和完整性"
                tasks = [
                    task.model_copy(update={"status": "pending", "error": ""}) if task.id in retry_ids else task
                    for task in tasks
                ]
                feedback_by_task = {task_id: feedback for task_id in retry_ids}
                memory.add_review_feedback(state["task_id"], retry_ids, review.reasons)
                emit("retrying", "审核未通过，正在补充证据并重试综合章节", 0.78, reasons=review.reasons)
                return {
                    "subtasks": tasks,
                    "report": report,
                    "review": review.model_dump(),
                    "retry_ids": retry_ids,
                    "feedback_by_task": feedback_by_task,
                    "review_cycles": cycles + 1,
                }
            return {"subtasks": tasks, "report": report, "review": review.model_dump(), "retry_ids": []}

        def finalize_node(state: ResearchState) -> ResearchState:
            report = state.get("report") or self._build_report(topic, state.get("subtasks", []))
            emit("finalizing", "正在保存 Markdown 报告和结构化结果", 0.95)
            self.settings.reports_dir.mkdir(parents=True, exist_ok=True)
            (self.settings.reports_dir / f"{task_id}.md").write_text(report, encoding="utf-8")
            return {"report": report}

        def route_after_review(state: ResearchState) -> str:
            review = state.get("review", {})
            return "execute" if state.get("retry_ids") and not review.get("passed", False) else "finalize"

        graph_builder = StateGraph(ResearchState)
        graph_builder.add_node("plan", plan_node)
        graph_builder.add_node("execute", execute_node)
        graph_builder.add_node("review", review_node)
        graph_builder.add_node("finalize", finalize_node)
        graph_builder.add_edge(START, "plan")
        graph_builder.add_edge("plan", "execute")
        graph_builder.add_edge("execute", "review")
        graph_builder.add_conditional_edges("review", route_after_review, {"execute": "execute", "finalize": "finalize"})
        graph_builder.add_edge("finalize", END)
        graph = graph_builder.compile()

        state: ResearchState = {
            "topic": topic,
            "task_id": task_id,
            "fast": fast,
            "memory_enabled": memory_enabled,
            "subtasks": [task.model_copy(deep=True) for task in preset_tasks] if preset_tasks else [],
            "report": "",
            "review": {},
            "retry_ids": [],
            "feedback_by_task": {},
            "review_cycles": 0,
            "errors": [],
        }
        try:
            final_state = graph.invoke(state, config={"recursion_limit": 30})
        except Exception as exc:
            emit("failed", f"研究任务失败：{exc}", 1.0, error=str(exc))
            raise
        tasks = final_state.get("subtasks", [])
        report = final_state.get("report", "")
        review = ReviewResult.model_validate(final_state.get("review", {"passed": False, "score": 0.0}))
        cited_ids = self._collect_citation_ids(tasks)
        citations = self.kb.get_by_ids(cited_ids)
        result = RunResult(
            task_id=task_id,
            topic=topic,
            memory_enabled=memory_enabled,
            report=report,
            citations=citations,
            completed=review.passed and all(task.status == "done" for task in tasks),
            review_score=review.score,
            review_reasons=review.reasons,
            degraded=bool(tasks) and any(task.status != "done" for task in tasks),
            elapsed_seconds=round(time.perf_counter() - started, 2),
            errors=[str(item) for item in final_state.get("errors", [])],
        )
        emit("completed", "研究完成", 1.0, result=result.model_dump(mode="json"))
        (self.settings.reports_dir / f"{task_id}.json").write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    @staticmethod
    def _execute_layer(
        *,
        executor: ExecutorAgent,
        topic: str,
        task_id: str,
        memory_enabled: bool,
        tasks: list[Subtask],
        ready: list[Subtask],
        feedback_by_task: dict[str, str],
        memory_contexts: dict[str, str],
    ) -> list[Subtask]:
        completed = {task.id: task for task in tasks if task.status == "done"}
        results: dict[str, Subtask] = {}
        workers = min(4, max(1, len(ready)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="research-executor") as pool:
            futures = {}
            for task in ready:
                missing = [dep for dep in task.dependencies if dep not in completed]
                if missing:
                    results[task.id] = task.model_copy(
                        update={
                            "status": "failed",
                            "attempts": task.attempts + 1,
                            "error": f"依赖任务未完成：{', '.join(missing)}",
                        }
                    )
                    continue
                dependency_results = (
                    [completed[dep].result for dep in task.dependencies if dep in completed]
                    if memory_enabled
                    else []
                )
                futures[
                    pool.submit(
                        executor.execute,
                        task_id=task_id,
                        topic=topic,
                        task=task,
                        dependency_results=dependency_results,
                        memory_enabled=memory_enabled,
                        memory_context=memory_contexts.get(task.id, ""),
                        feedback=feedback_by_task.get(task.id, ""),
                    )
                ] = task.id
            for future in as_completed(futures):
                task_id_value = futures[future]
                try:
                    results[task_id_value] = future.result()
                except Exception as exc:  # 单个子任务失败不能拖垮整个研究任务
                    original = next(task for task in ready if task.id == task_id_value)
                    results[task_id_value] = original.model_copy(
                        update={
                            "status": "failed",
                            "attempts": original.attempts + 1,
                            "error": str(exc),
                        }
                    )
        return [results.get(task.id, task) for task in tasks]

    @staticmethod
    def _weak_task_ids(tasks: list[Subtask]) -> list[str]:
        uncited = [task.id for task in tasks if task.status == "done" and not task.citations]
        return uncited or [task.id for task in tasks if task.status == "done"][:2]

    @staticmethod
    def _collect_citation_ids(tasks: list[Subtask]) -> list[str]:
        return list(dict.fromkeys(citation for task in tasks for citation in task.citations))

    def _build_report(self, topic: str, tasks: list[Subtask]) -> str:
        lines = [f"# {topic}：企业知识研究报告", "", "> 本报告由 research-agent 基于本地知识库自动生成；引用仅指向可回溯的证据切片。", ""]
        for task in tasks:
            if task.status == "done" and task.result:
                lines.extend([task.result.strip(), ""])
        lines.extend(["## 引用来源", ""])
        citations = list(dict.fromkeys(citation for task in tasks for citation in task.citations))
        if citations:
            for chunk in self.kb.get_by_ids(citations):
                lines.append(f"- [{chunk.chunk_id}] {chunk.title} — {chunk.source}")
        else:
            lines.append("- 无")
        return "\n".join(lines).strip() + "\n"