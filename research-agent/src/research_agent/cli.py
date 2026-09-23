from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from .agents import ExecutorAgent
from .config import get_settings
from .graph import ResearchAgent
from .llm import DeepSeekLLM, LLMError
from .rag import KnowledgeBase
from .storage import MemoryStore, StorageError, build_database

_MEMORY_CHOICES = {"on", "off"}


def _add_local_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local", action="store_true", help="使用内存存储后端，仅用于离线冒烟/CI")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-agent", description="企业知识研究助手 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="检查数据库、Redis、模型配置和知识库状态")
    _add_local_option(doctor)

    ingest = subparsers.add_parser("ingest", help="重建本地知识库索引")
    _add_local_option(ingest)
    ingest.add_argument("--dir", type=Path, default=None, help="文档目录，默认 knowledge_base")

    demo = subparsers.add_parser("demo", help="运行一次完整研究流程")
    _add_local_option(demo)
    demo.add_argument("--topic", required=True, help="研究主题")
    demo.add_argument("--memory", choices=sorted(_MEMORY_CHOICES), default="on")
    demo.add_argument("--fast", action="store_true", help="使用 5 个任务的快速计划")
    demo.add_argument("--task-id", default=None)

    ask = subparsers.add_parser("ask", help="基于已完成任务继续追问")
    _add_local_option(ask)
    ask.add_argument("--task-id", required=True)
    ask.add_argument("--question", required=True)

    retrieval_eval = subparsers.add_parser("eval-retrieval", help="计算关键词/向量/混合检索 Recall@k 和 MRR")
    _add_local_option(retrieval_eval)

    evaluation = subparsers.add_parser("eval-memory", help="运行开/关记忆消融实验")
    _add_local_option(evaluation)
    evaluation.add_argument("--limit", type=int, default=12)
    evaluation.add_argument("--memory", choices=["on", "off", "both"], default="both")
    evaluation.add_argument("--fast", action="store_true")
    subparsers.add_parser("report-metrics", help="把检索与记忆实验 JSON 汇总为 Markdown 验收报告")
    return parser


def ensure_knowledge_base(database: Any, kb: KnowledgeBase, *, reindex: bool = False) -> None:
    database.init_schema()
    if reindex or database.chunk_count() == 0:
        stats = kb.ingest_dir(get_settings().knowledge_base_dir)
        print(f"知识库已索引：{stats['documents']} 份文档，{stats['chunks']} 个切片")


def command_doctor(args: argparse.Namespace) -> int:
    del args
    settings = get_settings()
    database = build_database(settings)
    memory = MemoryStore(enabled=True, db=database, settings=settings)
    checks: list[tuple[str, bool, str]] = []
    try:
        database.ping()
        backend = settings.storage_backend
        checks.append((f"存储后端({backend})", True, "连接正常"))
        try:
            count = database.chunk_count()
            checks.append(("知识库", count > 0, f"{count} 个切片"))
        except StorageError:
            checks.append(("知识库", False, "表未初始化，请运行 ingest"))
    except StorageError as exc:
        checks.append((f"存储后端({settings.storage_backend})", False, str(exc)))
    try:
        memory.ping()
        checks.append(("Redis", True, "连接正常"))
    except Exception as exc:
        checks.append(("Redis", False, str(exc)))
    checks.append(("DeepSeek 配置", settings.llm_enabled, "已配置" if settings.llm_enabled else "缺少 API Key"))
    checks.append(("Embedding 配置", True, settings.embedding_model))
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def command_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    database = build_database(settings)
    kb = KnowledgeBase(database)
    directory = args.dir or settings.knowledge_base_dir
    database.init_schema()
    stats = kb.ingest_dir(directory)
    print(f"知识库已重建：{stats['documents']} 份文档，{stats['chunks']} 个切片")
    return 0


def command_demo(args: argparse.Namespace) -> int:
    settings = get_settings()
    database = build_database(settings)
    kb = KnowledgeBase(database)
    ensure_knowledge_base(database, kb)
    agent = ResearchAgent(settings, db=database, kb=kb)
    memory_enabled = args.memory == "on"
    print(f"开始研究：{args.topic}")
    result = agent.run(args.topic, memory_enabled=memory_enabled, fast=args.fast, task_id=args.task_id)
    memory = MemoryStore(enabled=memory_enabled, db=database, settings=settings)
    payload = result.model_dump()
    payload["created_at"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    payload["history"] = []
    memory.save_task(result.task_id, payload)
    print(f"任务完成：task_id={result.task_id}")
    print(f"记忆模式：{'开启' if memory_enabled else '关闭'}")
    print(f"审核结果：{'通过' if result.completed else '未通过'}，score={result.review_score:.2f}")
    print(f"耗时：{result.elapsed_seconds:.2f}s，引用：{len(result.citations)} 个证据切片")
    print(f"报告：{settings.reports_dir / (result.task_id + '.md')}")
    if result.errors:
        print("错误：" + "；".join(result.errors))
    return 0 if result.completed else 2


def command_ask(args: argparse.Namespace) -> int:
    settings = get_settings()
    database = build_database(settings)
    kb = KnowledgeBase(database)
    memory = MemoryStore(enabled=True, db=database, settings=settings)
    task = memory.load_task(args.task_id)
    if not task:
        result_path = settings.reports_dir / f"{args.task_id}.json"
        if result_path.exists():
            task = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            print(f"找不到任务：{args.task_id}", file=sys.stderr)
            return 2
    ensure_knowledge_base(database, kb)
    evidence = kb.search(args.question, limit=settings.retrieval_top_k, mode="hybrid")
    if not evidence:
        print("知识库中没有找到可引用证据，无法回答。")
        return 2
    context = ExecutorAgent._format_evidence(evidence)
    memory_context = memory.context(args.task_id, args.question) if task.get("memory_enabled", True) else ""
    llm = DeepSeekLLM(settings)
    system = "你是企业知识研究助手。只能依据给定证据回答，引用必须使用真实 chunk_id。"
    user = f"""
原研究主题：{task['topic']}
已有报告摘要：{task.get('report', '')[:2500]}
跨步骤记忆：
{memory_context or '无'}

追问：{args.question}

可用证据：
{context}

请用中文给出简洁答案，每个关键结论后使用 [chunk_id] 引用。证据不足时明确说明。
"""
    try:
        answer = llm.chat(system, user, temperature=0.1)
    except LLMError:
        answer = "模型暂不可用，以下为检索到的原始证据：\n\n" + "\n\n".join(
            f"- [{chunk.chunk_id}] {chunk.title}：{chunk.content[:300]}" for chunk in evidence
        )
    valid_ids = {chunk.chunk_id for chunk in evidence}
    cited = ExecutorAgent.extract_citations(answer, valid_ids)
    if not cited:
        answer += "\n\n引用：" + "、".join(f"[{chunk.chunk_id}]" for chunk in evidence[:2])
    history = list(task.get("history", []))
    history.append({"question": args.question, "answer": answer})
    task["history"] = history
    memory.save_task(args.task_id, task)
    print(answer)
    return 0


def command_eval_retrieval(args: argparse.Namespace) -> int:
    del args
    settings = get_settings()
    database = build_database(settings)
    kb = KnowledgeBase(database)
    ensure_knowledge_base(database, kb)
    dataset = _read_jsonl(settings.evaluation_dir / "retrieval_questions.jsonl")
    modes = ["keyword", "vector", "hybrid"]
    result: dict[str, Any] = {}
    for mode in modes:
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        for item in dataset:
            relevant = set(item["relevant_doc_ids"])
            hits = kb.search(item["question"], limit=settings.retrieval_top_k, mode=mode)
            retrieved_docs = [chunk.doc_id for chunk in hits]
            recalls.append(len(relevant.intersection(retrieved_docs)) / max(1, len(relevant)))
            rank = next((index for index, doc_id in enumerate(retrieved_docs, start=1) if doc_id in relevant), None)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        result[mode] = {
            "recall_at_k": round(statistics.fmean(recalls), 4) if recalls else 0.0,
            "mrr": round(statistics.fmean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
            "k": settings.retrieval_top_k,
            "questions": len(dataset),
        }
    output = settings.evaluation_dir / "results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "retrieval_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'模式':<10}{'Recall@k':>12}{'MRR':>10}")
    for mode in modes:
        print(f"{mode:<10}{result[mode]['recall_at_k']:>12.4f}{result[mode]['mrr']:>10.4f}")
    print(f"结果：{output / 'retrieval_metrics.json'}")
    return 0


def command_eval_memory(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings.retrieval_top_k = 3
    database = build_database(settings)
    kb = KnowledgeBase(database)
    ensure_knowledge_base(database, kb)
    topics = _read_jsonl(settings.evaluation_dir / "research_topics.jsonl")[: args.limit]
    memory_modes = [True, False] if args.memory == "both" else [args.memory == "on"]
    agent = ResearchAgent(settings, db=database, kb=kb)
    records: list[dict[str, Any]] = []
    for item in topics:
        plan = agent.planner.fallback_plan(item["topic"], count=6)
        for memory_enabled in memory_modes:
            result = agent.run(
                item["topic"],
                memory_enabled=memory_enabled,
                fast=args.fast,
                task_id=f"eval-{'on' if memory_enabled else 'off'}-{item['id']}",
                preset_tasks=[task.model_copy(deep=True) for task in plan],
            )
            records.append(
                {
                    "topic_id": item["id"],
                    "topic": item["topic"],
                    "memory": "on" if memory_enabled else "off",
                    "completed": result.completed,
                    "review_score": result.review_score,
                    "review_reasons": result.review_reasons,
                    "citation_count": len(result.citations),
                    "elapsed_seconds": result.elapsed_seconds,
                    "task_id": result.task_id,
                    "errors": result.errors,
                }
            )
            print(
                f"[{item['id']}] memory={'on' if memory_enabled else 'off'} "
                f"completed={result.completed} score={result.review_score:.2f}"
            )
    summary = _summarize_memory(records)
    output = settings.evaluation_dir / "results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "memory_ablation.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果：{output / 'memory_ablation.json'}")
    return 0


def command_report_metrics(args: argparse.Namespace) -> int:
    del args
    settings = get_settings()
    results = settings.evaluation_dir / "results"
    retrieval = json.loads((results / "retrieval_metrics.json").read_text(encoding="utf-8"))
    memory = json.loads((results / "memory_ablation.json").read_text(encoding="utf-8"))
    lines = [
        "# research-agent 数字验收报告",
        "",
        "## RAG 检索指标",
        "",
        "| 检索模式 | Recall@k | MRR | 问题数 | k |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("keyword", "vector", "hybrid"):
        item = retrieval[mode]
        lines.append(
            f"| {mode} | {item['recall_at_k']:.4f} | {item['mrr']:.4f} | {item['questions']} | {item['k']} |"
        )
    lines.extend(["", "## 记忆消融", "", "| 记忆模式 | 完成率 | 平均审核分 | 平均耗时(s) | 运行数 |", "|---|---:|---:|---:|---:|"])
    for mode in ("on", "off"):
        item = memory["summary"].get(mode)
        if item:
            lines.append(
                f"| {mode} | {item['completion_rate']:.4f} | {item['average_review_score']:.4f} | "
                f"{item['average_elapsed_seconds']:.2f} | {item['runs']} |"
            )
    delta = memory["summary"].get("completion_rate_delta")
    if delta is not None:
        lines.extend(
            [
                "",
                f"完成率差值：**{delta:+.4f}**。",
                "",
                "> 消融条件：同一主题使用固定 6 任务计划；每个子任务最多取 3 条当前检索证据；最终综合章节至少覆盖 4 个跨分支来源。",
            ]
        )
    failures = [item for item in memory["records"] if not item["completed"]]
    lines.extend(["", "## 未完成样本", ""])
    if failures:
        for item in failures:
            reasons = "; ".join(item.get("review_reasons", [])) or "未记录原因"
            lines.append(
                f"- `{item['topic_id']}` / memory={item['memory']} / score={item['review_score']:.2f} / {reasons}"
            )
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "> 说明：本报告由固定知识库、固定题目和同一模型配置实际运行生成；不以人工设定值代替实验输出。",
        ]
    )
    output = settings.reports_dir / "metrics-report.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"验收报告：{output}")
    return 0



def _summarize_memory(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in ("on", "off"):
        values = [item for item in records if item["memory"] == mode]
        if values:
            summary[mode] = {
                "runs": len(values),
                "completion_rate": round(sum(bool(item["completed"]) for item in values) / len(values), 4),
                "average_review_score": round(statistics.fmean(item["review_score"] for item in values), 4),
                "average_elapsed_seconds": round(statistics.fmean(item["elapsed_seconds"] for item in values), 2),
            }
    if "on" in summary and "off" in summary:
        summary["completion_rate_delta"] = round(
            summary["on"]["completion_rate"] - summary["off"]["completion_rate"],
            4,
        )
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到评估数据：{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "local", False):
        os.environ["STORAGE_BACKEND"] = "memory"
        get_settings.cache_clear()
    commands = {
        "doctor": command_doctor,
        "ingest": command_ingest,
        "demo": command_demo,
        "ask": command_ask,
        "eval-retrieval": command_eval_retrieval,
        "eval-memory": command_eval_memory,
        "report-metrics": command_report_metrics,
    }
    try:
        return commands[args.command](args)
    except (FileNotFoundError, StorageError, ValueError) as exc:
        print(f"执行失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())