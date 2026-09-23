from __future__ import annotations

import math
from pathlib import Path

from research_agent.config import Settings
from research_agent.embedding import Embedder
from research_agent.graph import ResearchAgent
from research_agent.llm import DeepSeekLLM
from research_agent.models import RetrievedChunk
from research_agent.rag import KnowledgeBase


class FakeDatabase:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def list_chunks(self) -> list[RetrievedChunk]:
        return self.chunks

    def vector_search(self, embedding: list[float], limit: int) -> list[RetrievedChunk]:
        scored = []
        for chunk in self.chunks:
            other = [float(ord(char) % 11) for char in chunk.content[: len(embedding)]]
            other += [0.0] * max(0, len(embedding) - len(other))
            dot = sum(a * b for a, b in zip(embedding, other, strict=True))
            norm_a = math.sqrt(sum(value * value for value in embedding)) or 1.0
            norm_b = math.sqrt(sum(value * value for value in other)) or 1.0
            scored.append((dot / (norm_a * norm_b), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk.model_copy(update={"score": score}) for score, chunk in scored[:limit]]

    def add_memory_item(self, **kwargs: object) -> None:
        del kwargs


def _fixture_chunks() -> list[RetrievedChunk]:
    texts = {
        "overview": "多 Agent 系统使用 Planner Executor Reviewer 分工，DAG 分层后同层任务可以并行执行。",
        "memory": "跨步骤摘要、反馈缓存和证据向量组成任务记忆，记忆消融需要比较完成率。",
        "rag": "RAG 使用 BM25 关键词检索和向量检索，通过 RRF 融合排名并计算 Recall 与 MRR。",
        "citation": "引用校验提取 chunk_id，与真实证据集合比对，无法回溯的引用应当删除。",
        "risk": "生产风险包括超时、重试、成本、权限、提示注入和失败不可见，需要明确降级边界。",
        "trend": "多 Agent 适合职责分离和独立审核，单一 Agent 加工作流在任务清晰时可能更便宜。",
    }
    return [
        RetrievedChunk(
            chunk_id=f"{name}::001",
            doc_id=name,
            title=name,
            source=f"local/{name}",
            content=content,
        )
        for name, content in texts.items()
    ]


def test_offline_graph_runs_end_to_end(tmp_path: Path) -> None:
    chunks = _fixture_chunks()
    settings = Settings(
        deepseek_api_key="",
        embedding_model="hash",
        embedding_dim=64,
        reports_dir=tmp_path,
        retrieval_top_k=4,
    )
    embedder = Embedder(settings)
    database = FakeDatabase(chunks)
    kb = KnowledgeBase(database, embedder)  # type: ignore[arg-type]
    kb._chunks_cache = chunks
    agent = ResearchAgent(
        settings,
        llm=DeepSeekLLM(settings, offline=True),
        db=database,  # type: ignore[arg-type]
        kb=kb,
    )
    result = agent.run("离线测试主题", memory_enabled=False, fast=True, task_id="offline-test")
    assert result.report
    assert result.citations
    assert "##" in result.report
    assert (tmp_path / "offline-test.md").exists()
