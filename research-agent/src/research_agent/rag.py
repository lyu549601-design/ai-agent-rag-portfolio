from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

import jieba
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from .embedding import Embedder, get_embedder
from .models import RetrievedChunk
from .storage import Database

SearchMode = Literal["keyword", "vector", "hybrid"]
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", flags=re.UNICODE)
_STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "是",
    "在",
    "对",
    "将",
    "中",
    "一个",
    "以及",
    "如何",
    "什么",
    "哪些",
}


class KnowledgeBase:
    """本地知识库：关键词 BM25、pgvector 向量检索和 RRF 混合检索。"""

    def __init__(
        self,
        db: Database | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.db = db or Database()
        self.embedder = embedder or get_embedder()
        self._chunks_cache: list[RetrievedChunk] | None = None

    def ingest_dir(self, directory: Path) -> dict[str, int]:
        chunks: list[RetrievedChunk] = []
        documents = 0
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf"}:
                continue
            text, metadata = self._read_document(path)
            if not text.strip():
                continue
            documents += 1
            chunks.extend(self._split_document(path, text, metadata))
        if not chunks:
            raise ValueError(f"目录中没有可解析文档：{directory}")
        embeddings = self.embedder.embed(chunk.content for chunk in chunks)
        self.db.replace_chunks(chunks, embeddings)
        self._chunks_cache = chunks
        return {"documents": documents, "chunks": len(chunks)}

    def search(self, query: str, *, limit: int = 5, mode: SearchMode = "hybrid") -> list[RetrievedChunk]:
        if mode == "keyword":
            return self._keyword_search(query, limit)
        if mode == "vector":
            return self.db.vector_search(self.embedder.embed_one(query), limit)
        keyword = self._keyword_search(query, max(limit * 2, 10))
        vector = self.db.vector_search(self.embedder.embed_one(query), max(limit * 2, 10))
        return self._reciprocal_rank_fusion(keyword, vector, limit)

    def document_count(self) -> int:
        return len({chunk.doc_id for chunk in self._all_chunks()})

    def all_chunk_ids(self) -> set[str]:
        return {chunk.chunk_id for chunk in self._all_chunks()}

    def get_by_ids(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        wanted = set(chunk_ids)
        return [chunk for chunk in self._all_chunks() if chunk.chunk_id in wanted]

    def _keyword_search(self, query: str, limit: int) -> list[RetrievedChunk]:
        chunks = self._all_chunks()
        if not chunks:
            return []
        tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = BM25Okapi(tokenized_corpus).get_scores(query_tokens)
        ranking = sorted(range(len(chunks)), key=lambda index: float(scores[index]), reverse=True)
        results: list[RetrievedChunk] = []
        for index in ranking[:limit]:
            score = float(scores[index])
            if score <= 0:
                continue
            results.append(chunks[index].model_copy(update={"score": score}))
        return results

    @staticmethod
    def _reciprocal_rank_fusion(
        keyword: list[RetrievedChunk],
        vector: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = {}
        by_id: dict[str, RetrievedChunk] = {}
        for ranking in (keyword, vector):
            for rank, chunk in enumerate(ranking, start=1):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (60 + rank)
                by_id[chunk.chunk_id] = chunk
        ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
        return [by_id[chunk_id].model_copy(update={"score": scores[chunk_id]}) for chunk_id in ordered]

    def _all_chunks(self) -> list[RetrievedChunk]:
        # ponytail: 演示知识库规模较小，全量缓存后现场计算 BM25；超过数万切片再改为数据库全文索引。
        if self._chunks_cache is None:
            self._chunks_cache = self.db.list_chunks()
        return self._chunks_cache

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = [token.strip().lower() for token in jieba.lcut(text)]
        return [token for token in tokens if token not in _STOPWORDS and _TOKEN_RE.search(token)]

    @staticmethod
    def _read_document(path: Path) -> tuple[str, dict[str, str]]:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        source = KnowledgeBase._extract_field(text, "来源") or str(path)
        title = KnowledgeBase._extract_field(text, "标题")
        if not title:
            heading = re.search(r"^#\s+(.+)$", text, flags=re.M)
            title = heading.group(1).strip() if heading else path.stem
        return text, {"title": title, "source": source}

    @staticmethod
    def _split_document(path: Path, text: str, metadata: dict[str, str]) -> list[RetrievedChunk]:
        doc_id = path.stem
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
        pieces: list[str] = []
        current = ""
        max_chars = 800
        overlap = 120
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 2 > max_chars:
                pieces.append(current)
                current = current[-overlap:] + "\n\n" + paragraph
            else:
                current = f"{current}\n\n{paragraph}".strip()
        if current:
            pieces.append(current)
        if not pieces:
            pieces = [text]

        chunks: list[RetrievedChunk] = []
        for index, content in enumerate(pieces):
            digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"{doc_id}::{index:03d}::{digest}",
                    doc_id=doc_id,
                    title=metadata["title"],
                    source=metadata["source"],
                    content=content,
                    metadata={"path": str(path)},
                )
            )
        return chunks

    @staticmethod
    def _extract_field(text: str, field: str) -> str | None:
        match = re.search(rf"^(?:[-*]\s*)?{re.escape(field)}[：:]\s*(.+)$", text, flags=re.M)
        return match.group(1).strip() if match else None
