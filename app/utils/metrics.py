from __future__ import annotations

from prometheus_client import Counter, Histogram

# Общие метрики сервиса. Используются и middleware, и pipeline.

requests_total = Counter(
    "pii_gateway_requests_total",
    "Total requests",
    ["method", "path", "status"],
)

request_duration = Histogram(
    "pii_gateway_request_duration_seconds",
    "Request duration in seconds",
    ["method", "path"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

entities_detected = Counter(
    "pii_gateway_entities_detected_total",
    "Total PII entities detected",
    ["entity_type"],
)

tokens_processed = Counter(
    "pii_gateway_tokens_processed_total",
    "Total tokens processed",
)


def observe_request(method: str, path: str, status: int, duration: float) -> None:
    """Фиксирует latency и счётчик запросов (RPS)."""
    requests_total.labels(method=method, path=path, status=str(status)).inc()
    request_duration.labels(method=method, path=path).observe(duration)


def observe_entities(entity_types: list[str]) -> None:
    """Увеличивает счётчик сущностей по типам."""
    for entity_type in entity_types:
        entities_detected.labels(entity_type=entity_type).inc()


def observe_tokens(count: int) -> None:
    """Увеличивает счётчик обработанных токенов (tokens per second)."""
    tokens_processed.inc(count)
