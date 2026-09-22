from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config.settings import settings
from app.core.pipeline import pipeline, PipelineError
from app.middleware.logging_middleware import RequestLoggingMiddleware, MetricsMiddleware
from app.models.request import (
    ProcessRequest,
    ProcessResponse,
    DetectOnlyResponse,
    MaskOnlyResponse,
    ErrorResponse,
    HealthResponse,
)
from app.services.rate_limiter import rate_limiter

if TYPE_CHECKING:
    from starlette.responses import Response

logger = logging.getLogger(__name__)

# Время старта для uptime
_start_time = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: инициализация и очистка ресурсов"""
    logger.info("Starting PII Masking Gateway...")
    await pipeline.initialize()
    await rate_limiter.initialize()
    logger.info("PII Masking Gateway started")
    yield
    logger.info("Shutting down PII Masking Gateway...")
    await rate_limiter.close()
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
        },
    )


@app.get("/metrics", tags=["metrics"])
async def metrics() -> "Response":
    """Prometheus metrics endpoint"""
    from starlette.responses import Response
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/process",
    response_model=ProcessResponse | DetectOnlyResponse | MaskOnlyResponse,
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
) -> ProcessResponse | DetectOnlyResponse | MaskOnlyResponse:
    """
    Обрабатывает текст: детектирует ПДн, маскирует перед LLM, демаскирует ответ.
    """
    request_id = getattr(http_request.state, "request_id", "unknown")

    # Rate limiting
    allowed = await rate_limiter.check(
        key=f"{request.system_id}:{http_request.client.host if http_request.client else 'unknown'}",
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=ErrorResponse(
                error="rate_limit_exceeded",
                message="Too many requests",
                request_id=request_id,
            ).model_dump(),
        )

    try:
        result = await pipeline.process(request)
        return result

    except PipelineError as e:
        logger.error(f"Pipeline error: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="pipeline_error",
                message=str(e),
                request_id=request_id,
            ).model_dump(),
        )

    except Exception as e:
        logger.error(f"Unexpected error: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="internal_error",
                message="Internal server error",
                request_id=request_id,
            ).model_dump(),
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Обработчик HTTP исключений"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
    )