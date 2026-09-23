from __future__ import annotations

import logging
import logging.config
import sys

from app.config.settings import settings
from app.utils.sanitizer import PIISanitizingFilter


def setup_logging() -> None:
    """Настраивает логирование с фильтром санитизации ПДн.

    Гарантирует, что исходные ПДн никогда не попадают в логи:
    каждый handler получает PIISanitizingFilter, который заменяет
    персональные данные на [REDACTED]. Идемпотентна — повторные вызовы
    не дублируют handlers.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if settings.log_format == "json":
        formatter: logging.Formatter = _json_formatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    root = logging.getLogger()
    root.setLevel(level)

    # Если корневой логгер уже настроен этим модулем — не дублируем handlers.
    if getattr(root, "_pii_gateway_configured", False):
        return

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(PIISanitizingFilter())

    # Убираем дефолтный lastResort handler, чтобы не было дублей
    root.handlers.clear()
    root.addHandler(console_handler)
    root._pii_gateway_configured = True

    # Логгеры приложения наследуют root handler.
    # Access-лог uvicorn дублирует request_end и на 1000 RPS занимает event loop.
    for name in ("app", "uvicorn", "uvicorn.error", "httpx", "redis"):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    access.setLevel(logging.WARNING)


def _json_formatter() -> logging.Formatter:
    try:
        from pythonjsonlogger.json import JsonFormatter

        return JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s %(module)s %(funcName)s %(lineno)d",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
                "message": "message",
            },
        )
    except ImportError:
        return logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
