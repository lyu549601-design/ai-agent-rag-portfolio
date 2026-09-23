from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """集中读取环境变量，避免各模块直接读取 .env。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    storage_backend: str = "postgres"
    postgres_url: str = "postgresql://postgres:password@localhost:5433/research_agent"
    redis_url: str = "redis://localhost:6380/0"

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    hf_endpoint: str = ""
    embedding_dim: int = 512

    retrieval_top_k: int = 5
    max_review_retries: int = 1
    llm_temperature: float = 0.2

    knowledge_base_dir: Path = PROJECT_ROOT / "knowledge_base"
    reports_dir: Path = PROJECT_ROOT / "reports"
    evaluation_dir: Path = PROJECT_ROOT / "evaluation"
    cache_dir: Path = PROJECT_ROOT / ".cache"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.deepseek_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings