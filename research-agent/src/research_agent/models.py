from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


TaskStatus = Literal["pending", "running", "done", "failed"]


class Subtask(BaseModel):
    """Planner 生成、Executor 消费的最小任务单元。"""

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    title: str = Field(min_length=2, max_length=80)
    objective: str = Field(min_length=5, max_length=500)
    dependencies: list[str] = Field(default_factory=list)

    status: TaskStatus = "pending"
    attempts: int = 0
    result: str = ""
    citations: list[str] = Field(default_factory=list)
    error: str = ""

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return value.strip()


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewResult(BaseModel):
    passed: bool
    score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    failed_task_ids: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    task_id: str
    topic: str
    memory_enabled: bool
    report: str
    citations: list[RetrievedChunk] = Field(default_factory=list)
    completed: bool
    review_score: float = Field(ge=0, le=1)
    review_reasons: list[str] = Field(default_factory=list)
    degraded: bool = False
    elapsed_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)