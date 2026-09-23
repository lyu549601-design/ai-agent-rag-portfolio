from __future__ import annotations

import json
import re
from collections import defaultdict

from .llm import DeepSeekLLM, LLMError
from .storage import MemoryStore
from .models import ReviewResult, RetrievedChunk, Subtask
from .rag import KnowledgeBase

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")


class PlannerAgent:
    """把主题拆成带依赖的 5–8 个研究子任务，并校验 DAG。"""

    def __init__(self, llm: DeepSeekLLM) -> None:
        self.llm = llm

    def plan(self, topic: str, *, fast: bool = False) -> list[Subtask]:
        if fast or not self.llm.available:
            return self.fallback_plan(topic, count=5 if fast else 6)
        system = (
            "你是企业研究 Planner。只输出 JSON，不要 Markdown。"
            "将研究主题拆成 5–8 个可独立执行、带依赖的子任务。"
        )
        user = f"""
研究主题：{topic}

输出格式：
{{
  "subtasks": [
    {{
      "id": "scope",
      "title": "研究范围",
      "objective": "明确需要回答的核心问题与边界",
      "dependencies": []
    }}
  ]
}}

要求：
1. 任务数量 5–8 个；
2. 至少形成 2 层依赖，同一层任务可并行；
3. id 使用英文小写字母、数字、下划线或短横线；
4. 不允许循环依赖、未知依赖或重复 id；
5. objective 要明确到可直接检索知识库并写成报告章节。
"""
        try:
            payload = self.llm.json(system, user)
            tasks = self._parse_tasks(payload)
            self._validate(tasks)
            return tasks
        except (LLMError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return self.fallback_plan(topic, count=5 if fast else 6)

    @staticmethod
    def fallback_plan(topic: str, *, count: int = 6) -> list[Subtask]:
        definitions = [
            ("scope", "研究范围与关键问题", f"明确“{topic}”的研究边界、核心问题和判断标准", []),
            ("landscape", "技术现状与主要路线", f"梳理“{topic}”的主要技术路线、代表方法和当前状态", ["scope"]),
            ("memory", "记忆与上下文机制", "分析 Agent 记忆、跨步骤摘要和反馈缓存的关键做法", ["scope"]),
            ("rag", "RAG 与引用可信", "分析文档解析、混合检索、引用真实性和知识更新机制", ["scope"]),
            (
                "evaluation",
                "评估、风险与生产挑战",
                "分析质量评估、成本延迟、安全边界和落地风险",
                ["landscape", "memory", "rag"],
            ),
            (
                "synthesis",
                "趋势判断与行动建议",
                "综合前文证据，给出发展趋势、选型建议和可执行路线",
                ["evaluation"],
            ),
        ]
        return [Subtask(id=item[0], title=item[1], objective=item[2], dependencies=item[3]) for item in definitions[:count]]

    @staticmethod
    def _parse_tasks(payload: dict) -> list[Subtask]:
        raw_tasks = payload.get("subtasks")
        if not isinstance(raw_tasks, list):
            raise ValueError("缺少 subtasks 数组")
        return [Subtask.model_validate(item) for item in raw_tasks]

    @staticmethod
    def _validate(tasks: list[Subtask]) -> None:
        if not 5 <= len(tasks) <= 8:
            raise ValueError("任务数量必须在 5–8 之间")
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("任务 id 重复")
        known = set(ids)
        for task in tasks:
            unknown = set(task.dependencies) - known
            if task.id in task.dependencies or unknown:
                raise ValueError(f"任务 {task.id} 含非法依赖：{unknown or task.id}")

        visiting: set[str] = set()
        visited: set[str] = set()
        graph = {task.id: task.dependencies for task in tasks}

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("存在循环依赖")
            if node in visited:
                return
            visiting.add(node)
            for dependency in graph[node]:
                visit(dependency)
            visiting.remove(node)
            visited.add(node)

        for task_id in ids:
            visit(task_id)


class ExecutorAgent:
    """单个研究子任务执行器：检索真实证据后生成带引用的章节。"""

    def __init__(self, kb: KnowledgeBase, llm: DeepSeekLLM, memory: MemoryStore) -> None:
        self.kb = kb
        self.llm = llm
        self.memory = memory

    def execute(
        self,
        *,
        task_id: str,
        topic: str,
        task: Subtask,
        dependency_results: list[str],
        memory_enabled: bool,
        memory_context: str = "",
        feedback: str = "",
    ) -> Subtask:
        query = f"{topic} {task.objective}"
        per_step_k = self.kb.embedder.settings.retrieval_top_k
        evidence = self.kb.search(query, limit=per_step_k, mode="hybrid")
        if not evidence:
            return task.model_copy(
                update={
                    "status": "failed",
                    "attempts": task.attempts + 1,
                    "error": "知识库没有检索到相关证据",
                }
            )

        dependency_context = "\n".join(dependency_results[-3:])
        memory_chunk_ids = set(_CITATION_RE.findall(memory_context))
        memory_evidence = self.kb.get_by_ids(list(memory_chunk_ids))
        combined_evidence = list({chunk.chunk_id: chunk for chunk in [*evidence, *memory_evidence]}.values())
        evidence_context = self._format_evidence(combined_evidence)
        try:
            result = self._generate(
                topic=topic,
                task=task,
                evidence_context=evidence_context,
                dependency_context=dependency_context,
                memory_context=memory_context,
                feedback=feedback,
            )
        except LLMError as exc:
            result = self._fallback_section(task, combined_evidence, str(exc))

        valid_ids = {chunk.chunk_id for chunk in combined_evidence}
        result = self.sanitize_citations(result, valid_ids)
        cited_ids = self.extract_citations(result, valid_ids)
        if not cited_ids:
            result = f"{result.rstrip()}\n\n引用：" + "、".join(f"[{chunk.chunk_id}]" for chunk in evidence[:2])
            cited_ids = [chunk.chunk_id for chunk in evidence[:2]]

        summary = self._summarize(task, result)
        if memory_enabled:
            self.memory.add_summary(task_id, task.id, summary)
            for chunk in combined_evidence:
                if chunk.chunk_id in cited_ids:
                    self.memory.add_evidence(task_id, chunk, task.id)

        return task.model_copy(
            update={
                "status": "done",
                "attempts": task.attempts + 1,
                "result": result.strip(),
                "citations": cited_ids,
                "error": "",
            }
        )

    def _generate(
        self,
        *,
        topic: str,
        task: Subtask,
        evidence_context: str,
        dependency_context: str,
        memory_context: str,
        feedback: str,
    ) -> str:
        system = (
            "你是企业知识研究 Executor。只能依据给定证据写作，不得补造事实或引用。"
            "输出中文 Markdown 章节，包含标题、分析正文和证据引用。"
        )
        user = f"""
总主题：{topic}
当前任务：{task.title}
任务目标：{task.objective}

上游结论：
{dependency_context or '无'}

跨步骤记忆：
{memory_context or '无'}

历史反馈：
{feedback or '无'}

可用证据：
{evidence_context}

写作要求：
1. 使用“## {task.title}”作为标题；
2. 200–500 字，明确回答任务目标；
3. 每个关键判断后必须引用证据，格式必须是 [真实 chunk_id]；
4. 只能引用上述证据里的 chunk_id，不得编造；
5. 信息不足时明确写出边界，不要把推测写成事实。
"""
        return self.llm.chat(system, user, temperature=0.15)

    @staticmethod
    def _format_evidence(evidence: list[RetrievedChunk]) -> str:
        blocks = []
        for chunk in evidence:
            blocks.append(
                f"[{chunk.chunk_id}]\n"
                f"标题：{chunk.title}\n"
                f"来源：{chunk.source}\n"
                f"内容：{chunk.content[:1200]}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _fallback_section(task: Subtask, evidence: list[RetrievedChunk], error: str) -> str:
        excerpts = " ".join(chunk.content[:220].replace("\n", " ") for chunk in evidence[:2])
        citations = "、".join(f"[{chunk.chunk_id}]" for chunk in evidence[:2])
        return f"## {task.title}\n\n{excerpts}（模型调用降级：{error}）\n\n引用：{citations}"

    @staticmethod
    def _summarize(task: Subtask, result: str) -> str:
        compact = re.sub(r"\s+", " ", result).strip()
        return f"{task.title}：{compact[:500]}"

    @staticmethod
    def extract_citations(text: str, valid_ids: set[str]) -> list[str]:
        return list(dict.fromkeys(match.group(1) for match in _CITATION_RE.finditer(text) if match.group(1) in valid_ids))

    @staticmethod
    def sanitize_citations(text: str, valid_ids: set[str]) -> str:
        return _CITATION_RE.sub(lambda match: match.group(0) if match.group(1) in valid_ids else "", text)


class ReviewerAgent:
    """先做确定性完整性/引用校验，再用 LLM 评估证据质量和完整性。"""

    def __init__(self, llm: DeepSeekLLM) -> None:
        self.llm = llm

    def review(
        self,
        *,
        topic: str,
        tasks: list[Subtask],
        report: str,
        known_chunk_ids: set[str],
    ) -> ReviewResult:
        reasons: list[str] = []
        failed = [task.id for task in tasks if task.status != "done"]
        if failed:
            reasons.append(f"未完成任务：{', '.join(failed)}")
        if len(report) < 400:
            reasons.append("报告正文过短")
        heading_count = len(re.findall(r"^##\s+", report, flags=re.M))
        if heading_count < max(3, len(tasks) // 2):
            reasons.append(f"报告章节不足：仅 {heading_count} 个")

        cited = set(_CITATION_RE.findall(report))
        invalid = sorted(cited - known_chunk_ids)
        if invalid:
            reasons.append(f"存在无法回溯的引用：{', '.join(invalid[:5])}")
        if not cited:
            reasons.append("报告没有任何可验证引用")

        done_tasks = [task for task in tasks if task.status == "done"]
        if len(done_tasks) >= 4 and done_tasks:
            final_doc_ids = {citation.split("::", 1)[0] for citation in done_tasks[-1].citations}
            prior_doc_ids = {
                citation.split("::", 1)[0]
                for task in done_tasks[:-1]
                for citation in task.citations
            }
            required = min(4, len(prior_doc_ids))
            if len(final_doc_ids) < required:
                reasons.append(
                    f"最终综合章节跨分支引用不足：覆盖 {len(final_doc_ids)} 个来源，至少需要 {required} 个"
                )

        deterministic_score = max(0.0, 1.0 - 0.2 * len(reasons))
        if reasons:
            return ReviewResult(
                passed=False,
                score=deterministic_score,
                reasons=reasons,
                failed_task_ids=failed,
            )

        if not self.llm.available:
            return ReviewResult(passed=True, score=0.8, reasons=["通过确定性校验（离线模式）"])

        system = "你是严格但公平的研究 Reviewer。只输出 JSON。"
        user = f"""
研究主题：{topic}

任务及章节：
{json.dumps([task.model_dump() for task in tasks], ensure_ascii=False)}

报告摘要：
{report[:8000]}

请检查完整性、逻辑一致性、引用充分性和事实边界，输出：
{{
  "passed": true,
  "score": 0.0,
  "reasons": ["..."],
  "failed_task_ids": []
}}

评分标准：0.8 以上可放行；引用缺失、章节空洞或依赖结论未整合应判不通过。
"""
        try:
            payload = self.llm.json(system, user, temperature=0.0)
            score = min(1.0, max(0.0, float(payload.get("score", deterministic_score))))
            return ReviewResult(
                passed=bool(payload.get("passed", score >= 0.8)),
                score=score,
                reasons=[str(item) for item in payload.get("reasons", [])],
                failed_task_ids=[str(item) for item in payload.get("failed_task_ids", []) if str(item) in {t.id for t in tasks}],
            )
        except (LLMError, ValueError, TypeError):
            return ReviewResult(passed=True, score=0.85, reasons=["LLM Judge 不可用，降级为确定性校验通过"])


def topological_layers(tasks: list[Subtask]) -> list[list[Subtask]]:
    """按依赖拓扑分层，同层任务交由上层并行执行。"""
    by_id = {task.id: task for task in tasks}
    indegree = {task.id: len(task.dependencies) for task in tasks}
    children: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        for dependency in task.dependencies:
            children[dependency].append(task.id)

    current = [task_id for task_id, degree in indegree.items() if degree == 0]
    layers: list[list[Subtask]] = []
    visited = 0
    while current:
        layer = sorted((by_id[task_id] for task_id in current), key=lambda item: item.id)
        layers.append(layer)
        visited += len(layer)
        next_ids: list[str] = []
        for task in layer:
            for child in children[task.id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    next_ids.append(child)
        current = next_ids
    if visited != len(tasks):
        raise ValueError("任务图无法拓扑排序，可能存在循环依赖")
    return layers