FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Системные зависимости для spacy и presidio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install uv && \
    uv pip install --system -e .

# Копируем код
COPY app/ ./app/

# Скачиваем модели spacy
RUN python -m spacy download ru_core_news_md && \
    python -m spacy download en_core_web_sm

# Создаем пользователя без прав
RUN useradd -m -u 1000 appuser && \
    mkdir -p /var/log/pii-gateway && \
    chown -R appuser:appuser /app /var/log/pii-gateway

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]