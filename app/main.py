from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config.settings import pii_rules
from app.core.payload_store import payload_store
from app.core.pipeline import PipelineError, pipeline
from app.middleware.logging_middleware import MetricsMiddleware, RequestLoggingMiddleware
from app.models.request import (
    ErrorResponse,
    HealthResponse,
    ProcessRequest,
    ProcessResponse,
)
from app.services.rate_limiter import rate_limiter
from app.utils.logging_setup import setup_logging

if TYPE_CHECKING:
    from starlette.responses import Response

# Настраиваем логирование с санитизацией ПДн до создания приложения,
# чтобы фильтр был активен для всех логгеров (включая uvicorn).
setup_logging()

logger = logging.getLogger(__name__)

# Время старта для uptime
_start_time = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: инициализация и очистка ресурсов"""
    # Идемпотентно гарантируем санитизацию ПДн в логах независимо от точки входа.
    setup_logging()
    logger.info("Starting PII Masking Gateway...")
    await pipeline.initialize()
    await rate_limiter.initialize()
    logger.info("PII Masking Gateway started")
    yield
    logger.info("Shutting down PII Masking Gateway...")
    await rate_limiter.close()
    await pipeline.close()
    logger.info("PII Masking Gateway stopped")


app = FastAPI(
    title="PII Masking Gateway",
    description="High-load service for protecting personal data before sending to LLM",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(MetricsMiddleware)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=time.monotonic() - _start_time,
        checks={
            "pipeline": True,
            "rate_limiter": True,
            "redis": payload_store.redis_available,
        },
    )


@app.get("/metrics", tags=["metrics"])
async def metrics() -> Response:
    """Prometheus metrics endpoint"""
    from starlette.responses import Response
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/process",
    response_model=ProcessResponse,
    tags=["processing"],
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def process_request(
    request: ProcessRequest,
    http_request: Request,
    x_system_id: str | None = Header(default=None, alias="X-System-Id"),
) -> ProcessResponse:
    """
    Обрабатывает текст по контракту AlfaSonar.

    Направление определяется по payload_id:
      - первый запрос с новым payload_id — маскирование;
      - второй запрос с тем же payload_id — демаскирование.
    """
    request_id = getattr(http_request.state, "request_id", "unknown")

    # Проверка системы до rate limiting и pipeline.
    if x_system_id is not None and not pii_rules.is_enabled(x_system_id):
        return JSONResponse(status_code=403, content={"error": "forbidden"})

    # Rate limiting по клиенту (host), чтобы один клиент держал 1000 RPS.
    # 429 с Retry-After отдаётся только при реальной перегрузке.
    client_key = http_request.client.host if http_request.client else "unknown"
    allowed = await rate_limiter.check(key=client_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=ErrorResponse(
                error="rate_limit_exceeded",
                message="Too many requests",
                request_id=request_id,
            ).model_dump(),
            headers={"Retry-After": "1"},
        )

    try:
        result = await pipeline.process(request.payload, request.payload_id, x_system_id)
        return ProcessResponse(result=result)

    except PipelineError:
        # Не логируем текст исключения — он может содержать ПДн.
        logger.error("pipeline_error", extra={"request_id": request_id})
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="pipeline_error",
                message="Internal processing error",
                request_id=request_id,
            ).model_dump(),
        ) from None

    except Exception:
        # Не логируем текст исключения — он может содержать ПДн.
        logger.error("internal_error", extra={"request_id": request_id})
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="internal_error",
                message="Internal server error",
                request_id=request_id,
            ).model_dump(),
        ) from None


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Обработчик HTTP исключений"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
        headers=exc.headers,
    )
