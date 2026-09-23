#!/bin/bash
# =============================================================================
# Команда для нагрузочного тестирования PII Masking Gateway
# =============================================================================
# 
# Этот скрипт запускает нагрузочное тестирование сервиса по контракту /process.
# Тестирование проверяет:
#   - Маскирование (прямой запрос с новым payload_id)
#   - Демаскирование (обратный запрос с тем же payload_id и маской)
#   - Latency ≤ 1s при RPS 1000 (в strict режиме)
#   - Отсутствие ошибок 429/5xx под нагрузкой
#
# Использование:
#   ./run_load_test.sh                          # базовый запуск (RPS=1000, duration=30s)
#   ./run_load_test.sh --rps 500 --duration 60  # кастомные параметры
#   ./run_load_test.sh --strict                 # строгий режим с проверкой latency
#
# Требования:
#   - Сервис должен быть запущен на http://127.0.0.1:8000 (или укажите --base-url)
#   - Установлены зависимости: pip install httpx
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Параметры по умолчанию
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RPS="${LOAD_RPS:-1000.0}"
DURATION="${LOAD_DURATION:-30.0}"
CONCURRENCY="${LOAD_CONCURRENCY:-200}"
TIMEOUT="${LOAD_TIMEOUT:-10.0}"
STRICT=""

# Парсинг аргументов командной строки
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --rps)
            RPS="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --strict)
            STRICT="--strict"
            shift
            ;;
        --help|-h)
            echo -e "${BLUE}Нагрузочное тестирование PII Masking Gateway${NC}"
            echo ""
            echo "Использование:"
            echo "  $0 [OPTIONS]"
            echo ""
            echo "Опции:"
            echo "  --base-url URL       Базовый URL сервиса (по умолчанию: $BASE_URL)"
            echo "  --rps N              Целевой RPS (по умолчанию: $RPS)"
            echo "  --duration N         Длительность теста в секундах (по умолчанию: $DURATION)"
            echo "  --concurrency N      Уровень конкурентности (по умолчанию: $CONCURRENCY)"
            echo "  --timeout N          Таймаут запроса в секундах (по умолчанию: $TIMEOUT)"
            echo "  --strict             Строгий режим с проверкой latency ≤ 1s"
            echo "  --help, -h           Показать эту справку"
            echo ""
            echo "Примеры:"
            echo "  $0 --rps 500 --duration 60"
            echo "  $0 --strict --rps 1000"
            echo "  BASE_URL=http://localhost:8080 $0"
            exit 0
            ;;
        *)
            echo -e "${RED}Неизвестный параметр: $1${NC}"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}Нагрузочное тестирование PII Masking Gateway${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo -e "${YELLOW}Параметры:${NC}"
echo "  Base URL:     $BASE_URL"
echo "  Target RPS:   $RPS"
echo "  Duration:     ${DURATION}s"
echo "  Concurrency:  $CONCURRENCY"
echo "  Timeout:      ${TIMEOUT}s"
echo "  Strict mode:  ${STRICT:-off}"
echo ""

# Проверка доступности сервиса
echo -e "${YELLOW}Проверка доступности сервиса...${NC}"
if ! curl -sS --max-time 5 "$BASE_URL/process" -X POST \
     -H "Content-Type: application/json" \
     -d '{"payload":"test","payload_id":"healthcheck"}' > /dev/null 2>&1; then
    echo -e "${RED}Сервис недоступен по адресу $BASE_URL${NC}"
    echo "Убедитесь, что сервис запущен:"
    echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"
    exit 1
fi
echo -e "${GREEN}✓ Сервис доступен${NC}"
echo ""

# Запуск нагрузочного теста
echo -e "${YELLOW}Запуск нагрузочного теста...${NC}"
echo -e "${BLUE}-----------------------------------------------------------------------------${NC}"

cd "$PROJECT_ROOT"
python -m scripts.load_check \
    --base-url "$BASE_URL" \
    --rps "$RPS" \
    --duration "$DURATION" \
    --concurrency "$CONCURRENCY" \
    --timeout "$TIMEOUT" \
    $STRICT

EXIT_CODE=$?

echo -e "${BLUE}-----------------------------------------------------------------------------${NC}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}=============================================================================${NC}"
    echo -e "${GREEN}✓ НАГРУЗОЧНЫЙ ТЕСТ ПРОЙДЕН${NC}"
    echo -e "${GREEN}=============================================================================${NC}"
else
    echo -e "${RED}=============================================================================${NC}"
    echo -e "${RED}✗ НАГРУЗОЧНЫЙ ТЕСТ НЕ ПРОЙДЕН${NC}"
    echo -e "${RED}=============================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Рекомендации:${NC}"
    echo "  1. Проверьте логи сервиса на наличие ошибок"
    echo "  2. Увеличьте количество воркеров: uvicorn ... --workers N"
    echo "  3. Проверьте настройки rate limiting"
    echo "  4. Для строгого режима убедитесь, что p95 latency ≤ 1s"
fi

exit $EXIT_CODE
