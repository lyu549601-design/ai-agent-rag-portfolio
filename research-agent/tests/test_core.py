from pathlib import Path

from research_agent.agents import ExecutorAgent, PlannerAgent, topological_layers
from research_agent.config import Settings
from research_agent.embedding import Embedder
from research_agent.models import Subtask
from research_agent.rag import KnowledgeBase


def test_fallback_plan_is_valid_dag() -> None:
    tasks = PlannerAgent.fallback_plan("测试主题")
    assert 5 <= len(tasks) <= 8
    layers = topological_layers(tasks)
    assert len(layers) >= 3
    assert {task.id for task in layers[0]} == {"scope"}


def test_citation_extraction_ignores_unknown_ids() -> None:
    text = "结论一 [doc-a::001]，结论二 [fake::999]，结论三 [doc-b::002]"
    assert ExecutorAgent.extract_citations(text, {"doc-a::001", "doc-b::002"}) == [
        "doc-a::001",
        "doc-b::002",
    ]


def test_hash_embedding_is_stable_and_normalized() -> None:
    settings = Settings(embedding_model="hash", embedding_dim=64)
    embedder = Embedder(settings)
    first = embedder.embed_one("多 Agent 记忆")
    second = embedder.embed_one("多 Agent 记忆")
    assert first == second
    assert len(first) == 64
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9


def test_document_split_keeps_source_fields() -> None:
    path = Path("demo-topic.md")
    text = "# 标题：样例\n标题: 示例文档\n来源: local/demo\n\n第一段内容。\n\n第二段内容。"
    metadata = {"title": "示例文档", "source": "local/demo"}
    chunks = KnowledgeBase._split_document(
        path, text, metadata
    )
    assert chunks
    assert chunks[0].doc_id == "demo-topic"
    assert chunks[0].source == "local/demo"


def test_unknown_dependency_rejected() -> None:
    tasks = [
        Subtask(id="a", title="任务A", objective="完成任务 A", dependencies=[]),
        Subtask(id="b", title="任务B", objective="完成任务 B", dependencies=["missing"]),
        Subtask(id="c", title="任务C", objective="完成任务 C", dependencies=[]),
        Subtask(id="d", title="任务D", objective="完成任务 D", dependencies=[]),
        Subtask(id="e", title="任务E", objective="完成任务 E", dependencies=[]),
    ]
    try:
        PlannerAgent._validate(tasks)
    except ValueError:
        return
    raise AssertionError("未知依赖必须被拒绝")