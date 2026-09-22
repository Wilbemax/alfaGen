Димастер, [22.09.2026, 21:03:20]:
ТЗ №3: независимый quality benchmark, настоящий load generator и чистая поставка
1. Роль
Ты работаешь как QA/performance engineer и инженер финальной поставки.
Рабочая ветка: task_C.
2. Цель
Сделать качество 95% и 1000 RPS объективно измеряемыми, а репозиторий — пригодным для чистой zip-поставки.
3. Почему задача нужна
147/147 pytest не является доказательством качества 95%.
Оригинальный evaluator:
сравнивает маску с эталоном через нормированное span-based расстояние Левенштейна;
проверяет точное восстановление исходника;
создаёт около 1000 RPS;
делает до двух retry;
учитывает Retry-After;
останавливается после пяти невалидных ответов подряд.
Текущий load_check.py выполняет запросы последовательно и ошибочно сообщает PASS при примерно 53 RPS.
4. Исходное состояние
Уже имеются:
scripts/load_check.py;
httpx;
API /process;
Prometheus metrics;
147 тестов;
ручной длинный round-trip;
Docker и Redis;
.gitignore, который уже исключает pyc, хотя 29 pyc ранее были добавлены в Git.
5. Разрешённые файлы
Изменять:
scripts/load_check.py;
README.
Создать:
scripts/quality_benchmark.py;
benchmarks/pii_quality.jsonl;
tests/unit/test_load_check.py;
tests/unit/test_quality_benchmark.py.
Удалить из Git:
только уже tracked __pycache__/*.pyc;
только уже tracked *.pyc.
.gitignore менять только если обнаружится реально отсутствующее правило. Сейчас необходимые правила уже есть.
6. Read-only файлы
Читать можно весь app/, конфигурацию и тесты.
Особенно:
app/main.py;
app/core/pipeline.py;
detector-файлы;
Masker;
tests/integration/test_api.py;
pyproject.toml;
Docker Compose.
7. Запрещённые изменения
Не менять runtime-код app/.
Не менять существующие detector unit tests.
Не менять API и формат маски.
Не добавлять зависимости.
Не исправлять detector под benchmark.
Не копировать только уже существующие unit cases.
Не включать реальные ПДн в dataset.
Не добавлять большой dataset или бинарные файлы.
Не заявлять 1000 RPS без измеренного результата.
Не включать bonus 2000 RPS в обязательный gate.
Не делать commit/push без команды.
8. Quality benchmark
8.1. Dataset
Создать компактный синтетический JSONL dataset.
Каждая запись:
{
  "id": "case-001",
  "text": "Дата рождения: 12.05.1990",
  "gold_entities": [
    {
      "type": "DATE_OF_BIRTH",
      "start": 15,
      "end": 25,
      "text": "12.05.1990"
    }
  ],
  "gold_mask": "Дата рождения: **********",
  "tags": ["positive", "date", "numeric"]
}
Требования:
только синтетические данные;
не менее 200 случаев;
все обязательные типы оригинального ТЗ;
не менее 8–10 positive вариантов на основной тип;
минимум 40 negative/false-positive traps;
минимум 30 mixed cases с несколькими типами;
различные падежи, регистры, пробелы, Unicode, границы строки;
минимум 30% случаев не должны быть прямыми копиями unit-тестов.
Обязательные категории:
PERSON;
DATE_OF_BIRTH;
PLACE_OF_BIRTH;
PASSPORT;
CITIZENSHIP;
PASSPORT_ISSUER;
PASSPORT_DEPT_CODE;
PASSPORT_ISSUE_DATE;
DRIVER_LICENSE;
ADDRESS;
EMAIL;
PHONE;
INN;
BANK_CARD;
CARD_CVV;
CARD_PIN;
CARD_HOLDER.
Negative traps:
историческая персона;
адрес банка;
обычная дата встречи;
год и статистическое число;
страна без контекста;
uppercase географическое название;
шестизначное число не как индекс;
номера, похожие одновременно на паспорт/ИНН/карту;
CVV/PIN без ключевого слова;
ORG/LOC без персонального контекста.
Dataset не передавать Developer A/B до фиксации их изменений. Это сохраняет характер blind validation.
8.2. Метрики
Считать exact entity key:
(entity_type, start, end)
Для каждого типа и в целом:
TP;
FP;
FN;
Precision;
Recall;
F1.
Вывести:
micro Precision/Recall/F1;
macro Precision/Recall/F1;
per-entity таблицу;
число invalid spans;
число случаев с нарушением text == source[start:end];
negative-case false-positive rate.
Дополнительно считать:
exact mask match rate;
normalized full-mask Levenshtein similarity:
1 - distance(predicted_mask, gold_mask) / max(lengths);
exact demask rate в HTTP-режиме.
Не называть full-mask metric точной официальной реализацией. В отчёте обозначить её как proxy: PDF не раскрывает точную формулу «span-based Levenshtein» и способ агрегирования 95%.
8.3. Режимы
Поддержать:
in-process mode:
CascadeDetector;
существующий Masker;
gold span/type metrics.
HTTP mode:
первый /process получает исходник;
второй — полученную маску с тем же ID;
сравниваются маска и точный исходник.
8.4. Quality gate
CLI должен поддерживать явные пороги.
Рекомендуемый финальный gate:
mean mask similarity proxy ≥ 0.95;
micro exact-span F1 ≥ 0.95;
macro F1 ≥ 0.90;
demask exact rate = 1.0;
invalid spans = 0;
invariant violations = 0;
negative false-positive rate ≤ 0.05.
Если оригинальная формула станет доступна, заменить proxy на неё до сдачи.
9. Load generator
9.1. Генерация нагрузки
Переделать load_check.py на open-loop scheduling:
планировать запросы по времени с target RPS;
не ждать завершения предыдущего запроса;
использовать bounded queue/semaphore;
настраивать max_connections, max_keepalive_connections, concurrency;
не создавать отдельный HTTP client на запрос;
ограничивать память через max in-flight;
отдельно проводить mask- и demask-фазы;
payload IDs уникальны в рамках прогона.
9.2. Retry semantics
максимум две повторные попытки;
учитывать Retry-After для 429;
429 не увеличивает и не сбрасывает consecutive-invalid counter;
успешный ответ сбрасывает counter;
network error, malformed response, 4xx кроме 429 и 5xx считаются invalid;
после пяти consecutive invalid запросов остановить прогон;
demask не запускать для mask case, который окончательно не прошёл.
9.3. Метрики
Печатать минимум:
target RPS;
scheduled requests;
started/completed requests;
achieved scheduling RPS;
achieved completion RPS;
success;
retried requests;
final 429;
4xx;
5xx;
network errors;
max consecutive invalid;
mask latency p50/p95/p99/max;
demask latency p50/p95/p99/max;
fraction latency > 1 second;
demask mismatches;
elapsed wall time;
peak in-flight.
9.4. PASS/FAIL
Strict PASS требует:
scheduled mask requests ≥ 95% от rps × duration;
achieved completed mask RPS ≥ 95% target;
mask p95 ≤ 1 секунда;
demask p95 ≤ 1 секунда;
0 demask mismatches;
0 final 5xx;
0 final 429 после retry;
max consecutive invalid < 5;
валидный JSON-контракт всех 200-ответов.
Вывести причины FAIL по отдельности.
Нельзя выдавать PASS только потому, что не было пяти 5xx подряд.
10. Unit tests
test_quality_benchmark.py
Проверить:
TP/FP/FN на искусственном примере;
micro и macro;
zero-division;
exact span/type;
Levenshtein: identical=1, fully different=0;
schema dataset;
gold text/offset invariant;
gold mask соответствует spans;
duplicate/overlapping gold entities отклоняются;
CLI gate возвращает правильный exit code.
test_load_check.py
Без реального сервера проверить через MockTransport/fake clock:
реально существует больше одного in-flight запроса;
scheduler создаёт ожидаемое число слотов;
semaphore ограничивает concurrency;
Retry-After;
максимум две retry;
правило пяти invalid;
429 не сбрасывает invalid counter;
success сбрасывает;
mismatch вызывает FAIL;
низкий achieved RPS вызывает FAIL;
высокий p95 вызывает FAIL;
корректный прогон возвращает 0.
Тесты не должны зависеть от wall-clock настолько, чтобы быть flaky.
11. Git hygiene и README
Удалить все tracked pyc/pycache.
README обновить только в своей области:
команды quality benchmark;
команды strict load test;
описание метрик;
честное замечание, что 1000 RPS подтверждается только конкретным отчётом;
уточнить, что active runtime settings идут из environment/.env, а pii_rules.yaml — из YAML;
не заявлять settings.yaml активным, если код его не читает;
сохранить инструкцию настройки систем не длиннее пяти предложений;
не переписывать README целиком.
12. Acceptance criteria
DONE, если:
benchmark покрывает все 17 типов;
dataset проходит schema/invariant validation;
рассчитываются все требуемые метрики;
load generator действительно конкурентный;
PASS зависит от фактического RPS и latency;
unit tests benchmark/load проходят;
полный pytest проходит;
tracked pyc отсутствуют;
README не содержит неподтверждённой декларации 1000 RPS;
git diff --check проходит;
runtime-файлы app/ не изменены.


Baseline может не пройти quality gate до merge task_A/task_B. Это допустимо; разработчик C обязан честно зафиксировать результат, а не подгонять dataset.
13. Команды проверки
python -m pytest tests/unit/test_quality_benchmark.py tests/unit/test_load_check.py -v
python -m pytest -v

python scripts/quality_benchmark.py `
  --dataset benchmarks/pii_quality.jsonl `
  --min-mask-similarity 0.95 `
  --min-micro-f1 0.95 `
  --min-macro-f1 0.90

python scripts/load_check.py `
  --base-url http://127.0.0.1:8000 `
  --rps 1000 `
  --duration 30 `
  --concurrency 2000 `
  --strict

git ls-files | Select-String -Pattern '(__pycache__|\.pyc$)'
git diff --check
git status --short
На Linux:
git ls-files | grep -E '(__pycache__|\.pyc$)'
Ожидаемый результат поиска tracked artifacts — пустой.
14. Отчёт разработчика
Верни:
описание dataset и распределение по типам;
формулы метрик;
оговорку о proxy для официальной метрики;
baseline и итоговые quality metrics;
target и achieved RPS;
p50/p95/p99;
ошибки/429/5xx/mismatches;
параметры машины и Docker;
список изменённых/удалённых файлов;
результаты тестов;
git diff --check;
git status --short;
ограничения benchmark и load test.
Commit и push не выполнять без отдельной команды.