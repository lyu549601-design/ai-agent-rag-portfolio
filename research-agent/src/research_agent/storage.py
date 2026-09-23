from __future__ import annotations

import asyncio
import json
import math
import uuid
from typing import Any

import asyncpg
import redis

from .config import PROJECT_ROOT, Settings, get_settings
from .embedding import Embedder
from .models import RetrievedChunk


class StorageError(RuntimeError):
    pass


class Database:
    """PostgreSQL/pgvector 访问层；演示规模下每次短连接，避免连接池复杂度。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def ping(self) -> bool:
        return self._run(self._ping())

    def init_schema(self) -> None:
        sql = (PROJECT_ROOT / "scripts" / "init_db.sql").read_text(encoding="utf-8")
        statements = [statement.strip() for statement in sql.split(";") if statement.strip()]
        self._run(self._execute_statements(statements))

    def chunk_count(self) -> int:
        return self._run(self._chunk_count())

    def replace_chunks(self, chunks: list[RetrievedChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量不一致")
        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.title,
                chunk.source,
                chunk.content,
                json.dumps(chunk.metadata, ensure_ascii=False),
                self._vector(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._run(self._replace_chunks(rows))

    def list_chunks(self) -> list[RetrievedChunk]:
        rows = self._run(self._list_chunks())
        return [self._chunk_from_row(row, score=0.0) for row in rows]

    def vector_search(self, embedding: list[float], limit: int) -> list[RetrievedChunk]:
        rows = self._run(self._vector_search(self._vector(embedding), limit))
        return [self._chunk_from_row(row, score=float(row["score"])) for row in rows]

    def add_memory_item(
        self,
        *,
        task_id: str,
        kind: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._run(
            self._add_memory_item(
                uuid.uuid4().hex,
                task_id,
                kind,
                content,
                json.dumps(metadata or {}, ensure_ascii=False),
                self._vector(embedding),
            )
        )

    def search_memory(self, task_id: str, embedding: list[float], limit: int) -> list[str]:
        return self._run(self._search_memory(task_id, self._vector(embedding), limit))

    def delete_task_memory(self, task_id: str) -> None:
        self._run(self._execute("DELETE FROM memory_items WHERE task_id = $1", task_id))

    async def _connect(self) -> asyncpg.Connection:
        try:
            return await asyncpg.connect(dsn=self.settings.postgres_url, timeout=5)
        except (OSError, asyncpg.PostgresError) as exc:
            raise StorageError(f"PostgreSQL 不可用：{exc}") from exc

    async def _ping(self) -> bool:
        conn = await self._connect()
        try:
            return (await conn.fetchval("SELECT 1")) == 1
        finally:
            await conn.close()

    async def _execute_statements(self, statements: list[str]) -> None:
        conn = await self._connect()
        try:
            for statement in statements:
                await conn.execute(statement)
        finally:
            await conn.close()

    async def _execute(self, sql: str, *args: Any) -> None:
        conn = await self._connect()
        try:
            await conn.execute(sql, *args)
        finally:
            await conn.close()

    async def _chunk_count(self) -> int:
        conn = await self._connect()
        try:
            return int(await conn.fetchval("SELECT count(*) FROM chunks"))
        finally:
            await conn.close()

    async def _replace_chunks(self, rows: list[tuple[Any, ...]]) -> None:
        conn = await self._connect()
        try:
            async with conn.transaction():
                await conn.execute("TRUNCATE TABLE chunks")
                await conn.executemany(
                    """
                    INSERT INTO chunks
                        (chunk_id, doc_id, title, source, content, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::vector)
                    """,
                    rows,
                )
        finally:
            await conn.close()

    async def _list_chunks(self) -> list[asyncpg.Record]:
        conn = await self._connect()
        try:
            return await conn.fetch(
                "SELECT chunk_id, doc_id, title, source, content, metadata FROM chunks ORDER BY chunk_id"
            )
        finally:
            await conn.close()

    async def _vector_search(self, vector: str, limit: int) -> list[asyncpg.Record]:
        conn = await self._connect()
        try:
            return await conn.fetch(
                """
                SELECT chunk_id, doc_id, title, source, content, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM chunks
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vector,
                limit,
            )
        finally:
            await conn.close()

    async def _add_memory_item(
        self,
        memory_id: str,
        task_id: str,
        kind: str,
        content: str,
        metadata: str,
        vector: str,
    ) -> None:
        conn = await self._connect()
        try:
            await conn.execute(
                """
                INSERT INTO memory_items
                    (memory_id, task_id, kind, content, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector)
                """,
                memory_id,
                task_id,
                kind,
                content,
                metadata,
                vector,
            )
        finally:
            await conn.close()

    async def _search_memory(self, task_id: str, vector: str, limit: int) -> list[str]:
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT content
                FROM memory_items
                WHERE task_id = $1
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                task_id,
                vector,
                limit,
            )
            return [row["content"] for row in rows]
        finally:
            await conn.close()

    @staticmethod
    def _run(coroutine: Any) -> Any:
        try:
            return asyncio.run(coroutine)
        except StorageError:
            raise
        except (OSError, asyncpg.PostgresError) as exc:
            raise StorageError(f"PostgreSQL 操作失败：{exc}") from exc

    @staticmethod
    def _vector(embedding: list[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"

    @staticmethod
    def _chunk_from_row(row: asyncpg.Record, *, score: float) -> RetrievedChunk:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            title=row["title"],
            source=row["source"],
            content=row["content"],
            score=score,
            metadata=metadata or {},
        )



class MemoryDatabase:
    """仅用于离线冒烟和 CI 的内存后端，不替代生产环境的 PostgreSQL/pgvector。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.chunks: list[RetrievedChunk] = []
        self.chunk_vectors: list[list[float]] = []
        self.memory_items: list[dict[str, Any]] = []

    def ping(self) -> bool:
        return True

    def init_schema(self) -> None:
        return None

    def chunk_count(self) -> int:
        return len(self.chunks)

    def replace_chunks(self, chunks: list[RetrievedChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量不一致")
        self.chunks = list(chunks)
        self.chunk_vectors = [list(vector) for vector in embeddings]

    def list_chunks(self) -> list[RetrievedChunk]:
        return list(self.chunks)

    def vector_search(self, embedding: list[float], limit: int) -> list[RetrievedChunk]:
        ranked = sorted(
            zip(self.chunks, self.chunk_vectors, strict=True),
            key=lambda item: self._cosine(embedding, item[1]),
            reverse=True,
        )
        return [
            chunk.model_copy(update={"score": self._cosine(embedding, vector)})
            for chunk, vector in ranked[:limit]
        ]

    def add_memory_item(
        self,
        *,
        task_id: str,
        kind: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.memory_items.append(
            {
                "task_id": task_id,
                "kind": kind,
                "content": content,
                "embedding": list(embedding),
                "metadata": metadata or {},
            }
        )

    def search_memory(self, task_id: str, embedding: list[float], limit: int) -> list[str]:
        items = [item for item in self.memory_items if item["task_id"] == task_id]
        ranked = sorted(items, key=lambda item: self._cosine(embedding, item["embedding"]), reverse=True)
        return [item["content"] for item in ranked[:limit]]

    def delete_task_memory(self, task_id: str) -> None:
        self.memory_items = [item for item in self.memory_items if item["task_id"] != task_id]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        norm_left = math.sqrt(sum(value * value for value in left)) or 1.0
        norm_right = math.sqrt(sum(value * value for value in right)) or 1.0
        return dot / (norm_left * norm_right)



class MemoryRedis:
    """仅用于本地演示的极小 Redis 替身；不提供跨进程持久化。"""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}

    def ping(self) -> bool:
        return True

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def rpush(self, key: str, value: str) -> None:
        self.lists.setdefault(key, []).append(value)

    def ltrim(self, key: str, start: int, end: int) -> None:
        values = self.lists.get(key, [])
        self.lists[key] = values[start:] if end == -1 else values[start : end + 1]

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self.lists.get(key, [])
        return values[start:] if end == -1 else values[start : end + 1]


class MemoryStore:
    """Redis 保存任务元数据和摘要，pgvector 保存可检索的记忆证据。"""

    def __init__(
        self,
        *,
        enabled: bool = True,
        db: Database | None = None,
        settings: Settings | None = None,
        redis_client: redis.Redis | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.enabled = enabled
        self.db = db or Database(self.settings)
        self.redis = redis_client or (
            MemoryRedis()
            if self.settings.storage_backend.lower() == "memory"
            else redis.Redis.from_url(self.settings.redis_url, decode_responses=True)
        )
        self.embedder = embedder or Embedder(self.settings)

    def ping(self) -> bool:
        return bool(self.redis.ping())

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None:
        self.redis.set(self._task_key(task_id), json.dumps(payload, ensure_ascii=False))

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        raw = self.redis.get(self._task_key(task_id))
        return json.loads(raw) if raw else None

    def add_summary(self, task_id: str, subtask_id: str, summary: str) -> None:
        if not self.enabled:
            return
        self.redis.rpush(self._summary_key(task_id), summary)
        self.redis.ltrim(self._summary_key(task_id), -20, -1)
        self.db.add_memory_item(
            task_id=task_id,
            kind="summary",
            content=summary,
            embedding=self._embed(summary),
            metadata={"subtask_id": subtask_id},
        )

    def add_feedback(self, task_id: str, feedback: str) -> None:
        if not self.enabled:
            return
        self.redis.rpush(self._feedback_key(task_id), feedback)
        self.redis.ltrim(self._feedback_key(task_id), -20, -1)
        self.db.add_memory_item(
            task_id=task_id,
            kind="feedback",
            content=feedback,
            embedding=self._embed(feedback),
        )

    def add_evidence(self, task_id: str, evidence: RetrievedChunk, subtask_id: str) -> None:
        if not self.enabled:
            return
        content = f"[{evidence.chunk_id}] {evidence.title}：{evidence.content}"
        self.db.add_memory_item(
            task_id=task_id,
            kind="evidence",
            content=content,
            embedding=self._embed(content),
            metadata={"subtask_id": subtask_id, "doc_id": evidence.doc_id},
        )

    def context(self, task_id: str, query: str, limit: int = 4) -> str:
        if not self.enabled:
            return ""
        recent = self.redis.lrange(self._summary_key(task_id), 0, -1)
        try:
            retrieved = self.db.search_memory(task_id, self._embed(query), limit)
        except StorageError:
            retrieved = []
        items = list(dict.fromkeys([*recent, *retrieved]))
        return "\n".join(f"- {item}" for item in items[-limit:])

    def add_review_feedback(self, task_id: str, task_ids: list[str], reasons: list[str]) -> None:
        text = f"复核未通过；任务={','.join(task_ids) or '整体'}；原因={'；'.join(reasons)}"
        self.add_feedback(task_id, text)

    def _embed(self, text: str) -> list[float]:
        return self.embedder.embed_one(text)

    @staticmethod
    def _task_key(task_id: str) -> str:
        return f"research:task:{task_id}"

    @staticmethod
    def _summary_key(task_id: str) -> str:
        return f"research:summary:{task_id}"

    @staticmethod
    def _feedback_key(task_id: str) -> str:
        return f"research:feedback:{task_id}"


def build_database(settings: Settings | None = None) -> Database | MemoryDatabase:
    resolved = settings or get_settings()
    if resolved.storage_backend.lower() == "memory":
        return MemoryDatabase(resolved)
    return Database(resolved)
