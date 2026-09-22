from __future__ import annotations
import logging
import time
import uuid
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.sanitizer import sanitize_text

if TYPE_CHECKING:
    from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware для безопасного логирования запросов.
    Гарантирует, что ПДн не попадут в логи.
    """

    def __init__(self, app: "ASGIApp") -> None:
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
            return response

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "request_error",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "error": str(e),
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware для сбора метрик Prometheus"""

    def __init__(self, app: "ASGIApp") -> None:
        super().__init__(app)
        self._init_metrics()

    def _init_metrics(self) -> None:
        from prometheus_client import Counter, Histogram

        self.requests_total = Counter(
            "pii_gateway_requests_total",
            "Total requests",
            ["method", "path", "status"],
        )
        self.request_duration = Histogram(
            "pii_gateway_request_duration_seconds",
            "Request duration in seconds",
            ["method", "path"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )
        self.entities_detected = Counter(
            "pii_gateway_entities_detected_total",
            "Total PII entities detected",
            ["entity_type"],
        )
        self.llm_requests = Counter(
            "pii_gateway_llm_requests_total",
            "Total LLM requests",
            ["status"],
        )
        self.llm_duration = Histogram(
            "pii_gateway_llm_duration_seconds",
            "LLM request duration in seconds",
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0),
        )

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
            self.requests_total.labels(method=method, path=path, status=str(response.status_code)).inc()
            self.request_duration.labels(method=method, path=path).observe(duration)
            return response
        except Exception:
            duration = time.monotonic() - start_time
            self.requests_total.labels(method=method, path=path, status="500").inc()
            self.request_duration.labels(method=method, path=path).observe(duration)
            raise