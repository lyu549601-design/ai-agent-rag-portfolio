from __future__ import annotations

import hashlib
import math
import os
import threading
from functools import lru_cache
from typing import Iterable

from .config import Settings, get_settings


class Embedder:
    """本地向量模型；hash 模式只用于离线测试和冒烟验证。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._lock = threading.Lock()

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        if self.settings.embedding_model == "hash":
            return [self._hash_embedding(text) for text in values]
        return self._fastembed(values)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _fastembed(self, texts: list[str]) -> list[list[float]]:
        if self.settings.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.settings.hf_endpoint)
        from fastembed import TextEmbedding
        with self._lock:
            if self._model is None:
                self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_dir = self.settings.cache_dir / "fastembed"
                local_model_cached = any(cache_dir.glob("*/model_optimized.onnx"))
                self._model = TextEmbedding(
                    model_name=self.settings.embedding_model,
                    cache_dir=str(cache_dir),
                    local_files_only=local_model_cached,
                )
            vectors = [vector.tolist() for vector in self._model.embed(texts)]
        self._validate_dimensions(vectors)
        return vectors

    def _hash_embedding(self, text: str) -> list[float]:
        dimension = self.settings.embedding_dim
        vector = [0.0] * dimension
        normalized = "".join(text.lower().split())
        tokens = [normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))]
        tokens.extend(normalized)
        for token in tokens or [normalized]:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            index = number % dimension
            vector[index] += 1.0 if number & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _validate_dimensions(self, vectors: list[list[float]]) -> None:
        expected = self.settings.embedding_dim
        actual = len(vectors[0]) if vectors else expected
        if actual != expected:
            raise ValueError(f"Embedding 维度不一致：配置为 {expected}，模型输出为 {actual}")


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()