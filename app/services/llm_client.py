from __future__ import annotations
import logging
import time
from typing import Any

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Ошибка при обращении к LLM"""


class LLMClient:
    """Асинхронный клиент для DeepSeek LLM"""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self.base_url = settings.llm_base_url
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout
        self.max_retries = settings.llm_max_retries
        self.retry_delay = settings.llm_retry_delay

    async def initialize(self) -> None:
        """Инициализация HTTP клиента"""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        logger.info("LLM client initialized")

    async def close(self) -> None:
        """Закрытие HTTP клиента"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Отправляет запрос к LLM и возвращает ответ.
        С ретраями при ошибках.
        """
        if not self._client:
            await self.initialize()

        assert self._client is not None

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503, 504):
                    # Ретраи для rate-limit и серверных ошибок
                    if attempt < self.max_retries - 1:
                        await self._sleep_with_backoff(attempt)
                        continue
                raise LLMClientError(f"LLM HTTP error: {e.response.status_code}") from e

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await self._sleep_with_backoff(attempt)
                    continue
                raise LLMClientError("LLM timeout") from e

            except Exception as e:
                last_error = e
                raise LLMClientError(f"LLM error: {e}") from e

        raise LLMClientError(f"LLM failed after {self.max_retries} attempts: {last_error}")

    async def _sleep_with_backoff(self, attempt: int) -> None:
        """Экспоненциальный backoff с джиттером"""
        import random
        delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 0.5)
        await __import__("asyncio").sleep(delay)


# Глобальный экземпляр
llm_client = LLMClient()