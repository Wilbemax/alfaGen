# PII Masking Gateway

Высоконагруженный сервис защиты персональных данных (PII Masking Gateway) для хакатона Альфа-Банка.

Сервис встраивается в цепочку: **Система-потребитель → Наш сервис → LLM (DeepSeek) → Наш сервис → Система-потребитель**

## Возможности

- **Детекция ПДн** через композитный детектор (Natasha + Microsoft Presidio + Regex)
- **Маскирование** ПДн токенами вида `[EMAIL_1]`, `[PASSPORT_2]`, `[CARD_3]` перед отправкой в LLM
- **Демаскирование** ответа LLM с восстановлением оригиналов
- **Гибкая конфигурация правил** для разных `system_id` (YAML)
- **Полное исключение ПДн из логов** через санитизацию
- **Rate limiting** (Redis / in-memory fallback)
- **Prometheus метрики**
- **Горизонтальное масштабирование** (stateless, request-scoped контекст)

## Архитектура

```
app/
├── main.py                    # FastAPI приложение
├── config/                    # Pydantic Settings + YAML конфигурация
├── models/                    # Pydantic модели запроса/ответа, request-scoped контекст
├── core/                      # Pipeline, Masker, Tokenizer
├── detectors/                 # Natasha, Presidio, Regex, Composite
├── services/                  # LLM client, Rate limiter
├── middleware/                # Безопасное логирование, метрики
└── utils/                     # Санитизация ПДн
```

## Быстрый старт

### Локально

```bash
# Установка зависимостей
pip install -e ".[dev]"

# Запуск
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker

```bash
cd docker
docker-compose up --build
```

## API

### POST /process

```json
{
  "system_id": "crm-system",
  "text": "Уважаемый Иван Иванов, ваш email: ivan@mail.ru",
  "mode": "mask_only"
}
```

Режимы:
- `full` — detect → mask → LLM → unmask
- `mask_only` — только маскирование
- `unmask_only` — только демаскирование
- `detect_only` — только детекция

### GET /health
Health check.

### GET /metrics
Prometheus метрики.

## Конфигурация

- `app/config/settings.yaml` — основные настройки
- `app/config/pii_rules.yaml` — правила ПДн для разных `system_id`
- `.env` — секреты и переменные окружения (см. `.env.example`)

## Тесты

```bash
pytest
```

## Безопасность

- Исходные ПДн никогда не попадают в логи (санитизация через `PIISanitizingFilter`)
- Маппинг оригиналов хранится только в request-scoped контексте и очищается после обработки
- Rate limiting защищает от перегрузки
- Все конфигурации через Pydantic Settings с валидацией