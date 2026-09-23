from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.utils.metrics import observe_request

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware:
    """
    Pure-ASGI middleware для безопасного логирования запросов.

    Гарантирует, что ПДн не попадут в логи. Реализован как чистый ASGI
    middleware (без BaseHTTPMiddleware), чтобы не гонять каждый запрос через
    отдельную задачу и не терять пропускную способность под нагрузкой.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._sensitive_paths = {"/process"}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        start_time = time.monotonic()
        path = scope.get("path", "")
        method = scope.get("method", "")

        status_code = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            duration_ms = (time.monotonic() - start_time) * 1000
            # Не логируем текст исключения — он может содержать ПДн.
            logger.error(
                "request_error",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise
        else:
            duration_ms = (time.monotonic() - start_time) * 1000
            # Логируем только метаданные, не тело
            logger.info(
                "request_end",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )


class MetricsMiddleware:
    """Pure-ASGI middleware для сбора метрик Prometheus (latency, RPS)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")

        # Не логируем метрики для /metrics
        if path == "/metrics":
            await self.app(scope, receive, send)
            return

        start_time = time.monotonic()
        status_code = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            duration = time.monotonic() - start_time
            observe_request(method, path, 500, duration)
            raise
        else:
            duration = time.monotonic() - start_time
            observe_request(method, path, status_code, duration)