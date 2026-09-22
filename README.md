# PII Masking Gateway

Высоконагруженный сервис защиты персональных данных (PII Masking Gateway) для хакатона Альфа-Банка.

Сервис встраивается в цепочку: **Система-потребитель → Наш сервис → LLM (DeepSeek) → Наш сервис → Система-потребитель**

## Возможности

- **Детекция ПДн** через каскад regex → Natasha → контекстный фильтр с расширяемыми правилами
- **Полное маскирование** ПДн звёздами той же длины, что и спан
- **Демаскирование** с восстановлением исходной строки побайтно по `payload_id`
- **Единый контракт** `POST /process` (`{payload, payload_id}` → `{result}`), направление определяется по `payload_id`
- **Полное исключение ПДн из логов** через санитизацию
- **Rate limiting** (Redis / in-memory fallback), 429 с `Retry-After` при перегрузке
- **Prometheus метрики**: latency, RPS, tokens per second, счётчик сущностей по типам
- **Горизонтальное масштабирование** (stateless, request-scoped контекст)

## Архитектура

```
app/
├── main.py                    # FastAPI приложение
├── config/                    # Pydantic Settings + YAML конфигурация
├── models/                    # Pydantic модели запроса/ответа
├── core/                      # Pipeline, Masker, PayloadStore
├── detectors/                 # Regex, Natasha, Cascade, ContextFilter
├── services/                  # Rate limiter
├── middleware/                # Безопасное логирование, метрики
└── utils/                     # Санитизация ПДн
```

## Быстрый старт

### Локально (один воркер)

```bash
# Установка зависимостей
pip install -e ".[dev]"

# Запуск одним воркером
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Docker

```bash
cd docker
docker-compose up --build
```

## API

### POST /process

Единый эндпоинт по контракту AlfaSonar. Обрабатывает и маскирование, и демаскирование. Направление определяется по `payload_id`:

- первый запрос с новым `payload_id` — **маскирование** (`payload` = исходная строка) → возвращает маску и запоминает соответствие;
- второй запрос с тем же `payload_id` и маской — **демаскирование** → возвращает исходную строку.

В ответе ПДн закрыты целиком звёздами той же длины, что и спан. Исходник хранится в хранилище до TTL в зашифрованном виде.

**Запрос**

```json
{
  "payload": "Клиент Иванов Иван Иванович, паспорт 4509 123456",
  "payload_id": "8a77d363c7c044b49b41d7b8a448243a"
}
```

**Ответ (200 OK)**

```json
{
  "result": "Клиент ********************, паспорт ***********"
}
```

Повторный запрос с тем же `payload_id` и исходником возвращает ту же маску (идемпотентность к ретраям). Запрос с маской и тем же `payload_id` возвращает исходную строку побайтно.

**Пример маскирования**

```bash
curl -sS -X POST http://127.0.0.1:8000/process \
  -H 'Content-Type: application/json' \
  -d '{"payload":"Клиент Иванов Иван Иванович, паспорт 4509 123456","payload_id":"demo-1"}'
# {"result":"Клиент ********************, паспорт ***********"}
```

**Пример демаскирования**

```bash
curl -sS -X POST http://127.0.0.1:8000/process \
  -H 'Content-Type: application/json' \
  -d '{"payload":"Клиент ********************, паспорт ***********","payload_id":"demo-1"}'
# {"result":"Клиент Иванов Иван Иванович, паспорт 4509 123456"}
```

**Ошибки**

- `403` — неизвестная или выключенная система (`X-System-Id`);
- `429` — перегрузка, в ответе заголовок `Retry-After`;
- `422` — невалидный запрос (нет `payload`/`payload_id`, лишние поля, превышен лимит длины в токенах);
- `500` — внутренняя ошибка сервиса.

### GET /health

Health check, включает состояние Redis (`checks.redis`).

### GET /metrics

Prometheus метрики: latency, RPS, tokens per second, счётчик сущностей по типам.

## Настройка систем

Правила находятся в `app/config/pii_rules.yaml`. У системы задаются поля `enabled`, `enabled_entity_types`, `demask_enabled`. Без заголовка `X-System-Id` работает профиль `checker`. Файл читается при старте сервиса. Новый тип добавляется детектором и строкой в списке типов.

## Конфигурация

Активные runtime-настройки читаются из переменных окружения и файла `.env` (см. `.env.example`). Правила ПДн читаются из YAML `app/config/pii_rules.yaml`. Файл `app/config/settings.yaml` кодом приложения не читается.

## Тесты

```bash
pytest
```

## Quality & Load Testing

### Quality benchmark

Независимый датасет `benchmarks/pii_quality.jsonl` — синтетические случаи, не копия unit-тестов. Скрипт считает exact-span TP/FP/FN, micro/macro Precision/Recall/F1, exact mask match, долю ложных срабатываний на negative-кейсах, число битых спанов и нарушений `text == source[start:end]`.

`mean_mask_similarity_proxy` — это `1 - levenshtein(predicted_mask, gold_mask) / max(len)`. Это proxy: PDF проверки не раскрывает точную формулу span-based Levenshtein и способ агрегации порога 95%.

**Quality in-process** (сравнивает `CascadeDetector` и `Masker` с gold-спанами):

```bash
python scripts/quality_benchmark.py \
  --dataset benchmarks/pii_quality.jsonl \
  --min-mask-similarity 0.95 \
  --min-micro-f1 0.95 \
  --min-macro-f1 0.90 \
  --max-negative-fp 0.05 \
  --min-demask-exact 1.0
```

**Quality HTTP** (маска и точное восстановление исходника измеряются двумя вызовами `POST /process` с одним `payload_id`):

```bash
python scripts/quality_benchmark.py \
  --dataset benchmarks/pii_quality.jsonl \
  --base-url http://127.0.0.1:8000 \
  --min-mask-similarity 0.95 \
  --min-micro-f1 0.95 \
  --min-macro-f1 0.90 \
  --max-negative-fp 0.05 \
  --min-demask-exact 1.0
```

Порог 95% подтверждается только напечатанным отчётом этого прогона. Датасет намеренно не подгоняется под текущие детекторы.

### Load check (strict)

`scripts/load_check.py` планирует маскирование open-loop с целевым RPS, одним HTTP-клиентом и ограничением in-flight. Затем отдельно демаскирует только успешные `payload_id`. До двух retry, `Retry-After` для 429, остановка после пяти невалидных ответов подряд. 429 не двигает счётчик невалидных ответов.

**Strict Load:**

```bash
python scripts/load_check.py \
  --base-url http://127.0.0.1:8000 \
  --rps 1000 \
  --duration 30 \
  --concurrency 2000 \
  --strict
```

PASS с флагом `--strict` требует фактический completed RPS и p95 latency, ноль расхождений демаскирования, ноль финальных 429/5xx и валидный JSON `{result}`.

> **1000 RPS считается достигнутым только при прохождении strict gate с printed report.** Цифра 2000 RPS не публикуется как достигнутая, пока конкретный strict-отчёт этого не показал.

### Глоссарий RPS

- **Target RPS** — целевая интенсивность планирования, задаётся флагом `--rps`. Это сколько запросов скрипт *пытается* запустить в секунду.
- **Scheduled RPS** — фактическая скорость, с которой скрипт *запланировал* слоты (запросы поставлены в очередь). Может быть ниже target при остановке из-за невалидных ответов.
- **Completed RPS** — фактическая скорость, с которой запросы *завершились* (получен ответ). Это главная метрика пропускной способности: она падает при медленных ответах и ограничении concurrency.
- **Success RPS** — скорость успешных запросов (валидный `{result}`). Учитывает только успехи, без 429/5xx/network/contract-ошибок.

Strict gate проверяет именно **completed RPS** (≥ 95% от target) и p95 latency, а не только scheduled.

### Предупреждение о storage

> **Multi-worker режим требует валидного `PAYLOAD_STORE_KEY` и доступного Redis.** В single-worker режиме допускается in-memory fallback, но шифрование не гарантируется.

## Безопасность

- Исходные ПДн никогда не попадают в логи (санитизация через `PIISanitizingFilter`)
- Исходник и маска хранятся в хранилище до TTL в зашифрованном виде (Fernet)
- Rate limiting защищает от перегрузки
- Все конфигурации через Pydantic Settings с валидацией