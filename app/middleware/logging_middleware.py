from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.metrics import observe_request

if TYPE_CHECKING:
    from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware для безопасного логирования запросов.
    Гарантирует, что ПДн не попадут в логи.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._sensitive_paths = {"/process"}

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.monotonic()
        path = request.url.path
        method = request.method

        # Логируем только безопасные данные (без тела запроса)
        logger.info(
            "request_start",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": request.client.host if request.client else None,
            },
        )

        try:
            response = await call_next(request)
            duration_ms = (time.monotonic() - start_time) * 1000

            # Логируем только метаданные, не тело
            logger.info(
                "request_end",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )
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
            return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware для сбора метрик Prometheus (latency, RPS)."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        method = request.method

        # Не логируем метрики для /metrics
        if path == "/metrics":
            return await call_next(request)

        start_time = time.monotonic()
        try:
            response = await call_next(request)
            duration = time.monotonic() - start_time
            observe_request(method, path, response.status_code, duration)
        except Exception:
            duration = time.monotonic() - start_time
            observe_request(method, path, 500, duration)
            raise
        else:
            return response
