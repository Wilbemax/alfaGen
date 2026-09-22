FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_WORKERS=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    NATASHA_DICTS_PATH=/app/natasha_dicts

WORKDIR /app

# Системные зависимости для рантайма (curl для healthcheck, libgomp1 для Natasha).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости и метаданные пакета
COPY pyproject.toml README.md ./

# Устанавливаем зависимости
RUN pip install --upgrade pip && \
    pip install uv && \
    uv pip install --system .

# Копируем код
COPY app/ ./app/

# Подготовка словарей Natasha/Navec/Slovnet (не блокирует сборку).
ARG NATASHA_DICTS_URL=""
RUN if [ -n "$NATASHA_DICTS_URL" ]; then \
        mkdir -p /app/natasha_dicts && \
        curl -fsSL "$NATASHA_DICTS_URL" -o /tmp/natasha_dicts.tar.gz && \
        tar -xzf /tmp/natasha_dicts.tar.gz -C /app/natasha_dicts && \
        rm -f /tmp/natasha_dicts.tar.gz; \
    fi

# Неблокирующая проверка импортов Natasha-зависимостей.
RUN python -c "import natasha, navec, slovnet" || true

# Создаем пользователя без прав
RUN useradd -m -u 1000 appuser && \
    mkdir -p /var/log/pii-gateway && \
    chown -R appuser:appuser /app /var/log/pii-gateway

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]