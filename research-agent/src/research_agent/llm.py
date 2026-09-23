from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

from .config import Settings, get_settings


class LLMError(RuntimeError):
    pass


class DeepSeekLLM:
    """DeepSeek 的 OpenAI 兼容封装，保留替换其他模型的空间。"""

    def __init__(self, settings: Settings | None = None, *, offline: bool = False) -> None:
        self.settings = settings or get_settings()
        self.offline = offline or not self.settings.llm_enabled
        self._client: OpenAI | None = None

    @property
    def available(self) -> bool:
        return not self.offline

    def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        if self.offline:
            raise LLMError("离线模式没有可用的对话模型")

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._get_client().chat.completions.create(
                    model=self.settings.deepseek_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.settings.llm_temperature if temperature is None else temperature,
                    response_format={"type": "json_object"} if json_mode else None,
                )
                content = response.choices[0].message.content
                if not content:
                    raise LLMError("模型返回了空内容")
                return content.strip()
            except Exception as exc:  # SDK 异常类型较多，统一做有限重试
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"DeepSeek 调用失败：{last_error}") from last_error

    def json(self, system: str, user: str, *, temperature: float | None = None) -> dict[str, Any]:
        raw = self.chat(system, user, json_mode=True, temperature=temperature)
        return self._parse_json(raw)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
                timeout=90.0,
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S | re.I)
        if fence:
            cleaned = fence.group(1).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start < 0 or end <= start:
                raise LLMError(f"无法解析 JSON：{raw[:200]}") from exc
            parsed = json.loads(cleaned[start : end + 1])
        if not isinstance(parsed, dict):
            raise LLMError("模型 JSON 顶层必须是对象")
        return parsed