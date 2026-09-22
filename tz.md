# ТЗ №2 — PayloadStore, Redis, multi-worker и безопасный storage lifecycle

Рекомендуемая ветка: `task_storage`.

## 1. Роль разработчика

Backend/Infrastructure engineer с опытом Redis, async Python, шифрования и multi-worker deployment.

## 2. Главная цель

Сделать storage lifecycle корректным и доказуемым:

`configuration → Fernet → Redis → worker policy → ready state`.

При нескольких workers Redis и шифрование обязательны; process-local fallback невозможен. Два независимых экземпляра Pipeline/PayloadStore должны выполнять точный mask → demask round-trip через общее Redis-хранилище.

## 3. Почему это нужно

Текущий BLOCKER:

- multi-worker check выполняется до чтения `PAYLOAD_STORE_KEY`;
- до создания Fernet;
- до подключения и ping Redis;
- поэтому `UVICORN_WORKERS > 1` немедленно получает `RuntimeError`, даже если конфигурация правильная.

Дополнительные проблемы:

- пустой ключ приводит к plaintext memory storage;
- Redis runtime error тихо переключает worker на local memory;
- такой fallback ломает cross-worker round-trip;
- нет теста двух независимых workers/stores;
- TTL и NX проверены не полностью;
- `payload_id` не изолирован по `system_id`;
- Docker default допускает запуск без encryption key;
- README обещает более сильные гарантии, чем default runtime.

## 4. Scope

Разрешено изменять только:

- `app/core/payload_store.py`
- `app/core/pipeline.py`
- `app/config/settings.py`
- `docker/docker-compose.yml`
- `.env.example`
- `tests/unit/test_payload_store.py`
- `tests/unit/test_pipeline.py`

Разрешено создать:

- `tests/integration/test_storage_roundtrip.py`

Если требуется отдельный тип storage-ошибки, он должен находиться в `app/core/payload_store.py`, а не в новом общем модуле.

## 5. Read-only files

Можно читать, но нельзя менять:

- `app/main.py`
- `app/models/request.py`
- `app/models/pii.py`
- `app/core/masker.py`
- все `app/detectors/*`
- `app/services/rate_limiter.py`
- `app/utils/sanitizer.py`
- `app/utils/logging_setup.py`
- `app/middleware/logging_middleware.py`
- `app/config/pii_rules.yaml`
- `tests/integration/test_api.py`
- `scripts/quality_benchmark.py`
- `scripts/load_check.py`
- `benchmarks/pii_quality.jsonl`
- `README.md`
- `pyproject.toml`
- `tz.md`
- detection- и benchmark-тесты.

## 6. Forbidden scope

Запрещено:

- менять API `POST /process`;
- менять request/response schema;
- менять правила детекции;
- менять Masker;
- менять benchmark dataset;
- добавлять зависимости;
- переписывать rate limiter;
- логировать original, mask, entity text, encryption key или raw payload_id;
- хардкодить Fernet key;
- автоматически генерировать новый key при каждом worker startup;
- тихо переходить на memory в multi-worker;
- использовать разные storage keys для разных workers;
- ослаблять idempotency/NX;
- менять README — его единственный writer в этом спринте ТЗ №3;
- делать commit/push без команды.

## 7. Технические требования

### 7.1. Явная policy конфигурации

Добавить настройку:

`PAYLOAD_STORE_ALLOW_MEMORY_FALLBACK`

Рекомендуемый контракт:

- application default может разрешать fallback для локальной single-worker разработки, чтобы сохранить существующий developer flow;
- Docker/production Compose обязан устанавливать `false`;
- при `UVICORN_WORKERS > 1` значение этого флага игнорируется: Redis+Fernet обязательны всегда;
- `.env.example` объясняет назначение флага без реального секрета.

Не использовать неактивный `settings.yaml`.

### 7.2. Правильный initialize lifecycle

Порядок:

1. Прочитать worker count и fallback policy.
2. Прочитать `PAYLOAD_STORE_KEY`.
3. Если key задан — проверить его как Fernet key.
4. Создать Fernet.
5. Создать Redis client.
6. Выполнить `PING`.
7. Только после этого определить режим:
   - encrypted Redis ready;
   - разрешённый local-dev memory fallback;
   - fatal configuration/dependency error.
8. Установить согласованное ready/state поле.

Multi-worker:

- missing key → startup error;
- invalid key → startup error;
- Redis unavailable → startup error;
- Redis ping failure → startup error;
- успешный key+Redis → startup succeeds;
- ошибка не должна содержать key, original или mask.

Single-worker:

- key+Redis → encrypted Redis;
- fallback разрешён и Redis/key отсутствует → local memory с одним безопасным warning;
- fallback запрещён → startup error;
- invalid key при явно заданном key не должен молча трактоваться как «ключ отсутствует».

Initialize остаётся идемпотентным.

### 7.3. Runtime Redis failure policy

Для strict/production или multi-worker режима:

- `put/get` Redis failure не переключает worker на local memory;
- операция завершается контролируемой storage/pipeline error;
- сервис не создаёт divergent process-local запись;
- ошибка не раскрывает данные.

Для явно разрешённого single-worker local-dev fallback:

- допускается локальное хранение;
- режим должен быть предсказуем и протестирован;
- не смешивать частично успешную Redis-запись и локальную запись без формального правила.

Рекомендуется зафиксировать storage mode при initialize и не менять его скрыто на каждом запросе.

### 7.4. Encryption и Redis representation

Сохранить и проверить:

- Fernet;
- key только из environment/settings;
- Redis value содержит только ciphertext;
- original и masked text не присутствуют в Redis value в UTF-8;
- Redis key не содержит raw payload_id;
- key использует SHA-256;
- `SET NX`;
- TTL задаётся через `EX`;
- повторный `put` не перезаписывает запись;
- decrypt failure не приводит к возврату мусора или local stale record.

### 7.5. Namespace по system_id

Устранить cross-system collision:

- storage identity должна включать нормализованный system scope;
- запрос без `X-System-Id` использует стабильный scope, например `checker`;
- один `payload_id` в разных system scopes не должен читать или перезаписывать чужую запись;
- raw system ID и payload ID не должны попадать в Redis key или лог без хеширования;
- mask и demask обязаны использовать один и тот же scope;
- не менять внешний API-контракт.

Возможная сигнатура:

- `put(scope, payload_id, record)`
- `get(scope, payload_id)`

или эквивалентный внутренний key object.

### 7.6. Pipeline testability и два worker-like экземпляра

Разрешён минимальный dependency injection:

- `Pipeline` может принимать `PayloadStore` через constructor;
- default остаётся глобальный production store;
- существующие callers не меняются.

Это нужно для теста:

1. Pipeline A маскирует original.
2. Store A пишет encrypted record в общий fake/shared Redis.
3. Pipeline B имеет другой объект PayloadStore.
4. Store B читает ту же Redis-запись.
5. Pipeline B получает mask с тем же payload_id/scope.
6. Возвращается исходная строка посимвольно.

Нельзя доказывать cross-worker только повторным вызовом одного глобального объекта.

### 7.7. Docker policy

`docker/docker-compose.yml`:

- production-like API не запускается без явно переданного `PAYLOAD_STORE_KEY`;
- key не имеет default literal;
- `PAYLOAD_STORE_ALLOW_MEMORY_FALLBACK=false`;
- Redis остаётся `noeviction`;
- TTL больше времени load test;
- worker count читается из `UVICORN_WORKERS`;
- после исправления должна поддерживаться конфигурация нескольких workers;
- секрет не печатается и не записывается в image.

Не менять load generator.

### 7.8. Logging и sanitizer

Сохранить:

- raw payload_id отсутствует в логах;
- original/mask отсутствуют в логах;
- ciphertext не логируется;
- key не логируется;
- exception detail с Redis не должен включать command arguments/value;
- entity text не логируется;
- pipeline продолжает логировать только безопасные типы и counts.

## 8. Regression requirements

Не сломать:

- mask → demask для default checker;
- повтор исходника возвращает ту же маску;
- чужой текст с известным payload_id не перезаписывает пару;
- `demask_enabled=false`;
- entity limit;
- allowed entity types;
- payload_id sanitization;
- exact restored string;
- TTL local store;
- NX semantics;
- Redis hash key;
- полный single-worker test flow;
- API contract;
- quality benchmark и detector output.

## 9. Новые тесты

`test_payload_store.py`:

- multi-worker + valid Fernet key + успешный fake Redis ping → initialize success;
- multi-worker + missing key → RuntimeError;
- multi-worker + invalid key → RuntimeError;
- multi-worker + Redis unavailable → RuntimeError;
- multi-worker runtime `put` failure → error, local record не создаётся;
- multi-worker runtime `get` failure → error, local record не читается;
- single-worker + explicitly allowed fallback → memory works;
- single-worker + fallback disabled + missing key → startup error;
- invalid explicit key не деградирует в memory;
- ciphertext не содержит original/mask;
- raw payload_id и raw system ID не входят в Redis key;
- одинаковый payload_id в двух system scopes создаёт разные keys;
- TTL передаётся точно;
- NX предотвращает overwrite;
- decrypt failure возвращает контролируемую ошибку;
- initialize идемпотентен;
- close корректно сбрасывает state.

`test_pipeline.py` / новый integration test:

- два разных PayloadStore с одним shared fake Redis;
- mask через Pipeline A;
- demask через Pipeline B;
- exact Unicode/Cyrillic round-trip;
- тот же payload_id в другом system scope не раскрывает original;
- retry original через другой store возвращает ту же mask;
- production storage error преобразуется в безопасный `PipelineError`;
- логи не содержат original/mask/raw payload_id/key.

Fake Redis должен моделировать:

- `ping`;
- shared data;
- `SET NX EX`;
- `GET`;
- `aclose`;
- при необходимости проверяемый TTL metadata.

Не добавлять Docker/testcontainers dependency.

## 10. Команды проверки

Scoped:

```bash
python -m pytest tests/unit/test_payload_store.py tests/unit/test_pipeline.py -v
python -m pytest tests/integration/test_storage_roundtrip.py -v
```

Regression:

```bash
python -m pytest tests/integration/test_api.py -v
python -m pytest -v
```

Docker startup после реализации:

```powershell
$env:PAYLOAD_STORE_KEY = "<valid-Fernet-key>"
$env:UVICORN_WORKERS = "2"
docker compose -f docker/docker-compose.yml up --build
```

Проверить:

```bash
curl http://127.0.0.1:8000/health
```

Cross-worker runtime допускается проверять серией mask/demask запросов с несколькими уникальными payload_id. Финальный 1000 RPS выполняется после merge ТЗ №3.

Контроль:

```bash
git diff -- benchmarks/pii_quality.jsonl
git diff --check
git status --short
```

## 11. Acceptance criteria

- valid key + available Redis + workers=2 успешно инициализируются;
- missing/invalid key при workers>1 вызывает fail-fast;
- unavailable Redis при workers>1 вызывает fail-fast;
- production Docker не допускает memory fallback;
- runtime Redis failure в multi-worker не создаёт local record;
- Redis values зашифрованы;
- TTL=3600 либо актуальное настроенное значение;
- NX/idempotency сохранены;
- два независимых store/pipeline объекта выполняют exact cross-store round-trip;
- system scope изолирует одинаковые payload_id;
- raw payload_id, original, mask и key отсутствуют в логах/Redis keys;
- API не изменён;
- полный pytest: `203 + новые тесты`, 0 failed;
- benchmark dataset не изменён;
- изменения ограничены scope.

## 12. Definition of Done

Разработчик возвращает:

- диаграмму/описание итогового lifecycle;
- root cause прежнего pre-initialization check;
- выбранную memory fallback policy;
- поведение для каждой комбинации workers/key/Redis/fallback;
- список изменённых файлов;
- новые тесты и их результаты;
- доказательство encrypted value;
- доказательство cross-store round-trip;
- Docker startup result с несколькими workers;
- известные ограничения;
- полный pytest;
- `git diff --check`;
- `git status --short`;
- инструкции для ТЗ №3, какие environment variables документировать.

Commit и push — только по отдельной команде.

AIL: completion RPS 282 < 950, mask p95 и demask p95 больше 1 с. 1000 RPS этим прогоном не подтверждены.

Контрольный strict-прогон 100 RPS × 3 с, concurrency 50, прошёл: completion RPS 100.14, mask p95 0.018 с, demask p95 0.150 с, mismatches 0.

Машина: darwin arm64, 10 CPU, Python 3.14.5. Docker Compose в репозитории задаёт API и Redis 7 с `maxmemory 1gb` и `noeviction`; этот замер шёл против локального процесса без Redis.

##
