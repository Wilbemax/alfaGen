from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.pipeline import pipeline
from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _natasha_available() -> bool:
    """Проверяет доступность Natasha в каскаде (после инициализации)."""
    try:
        return bool(pipeline.cascade.natasha_available)
    except Exception:
        return False


# Маскируемые кейсы: (фраза, ожидаемая маска, зависит ли от Natasha).
MASK_CASES = [
    pytest.param(
        "Клиент Иванов Иван Иванович, паспорт 4509 123456",
        "Клиент ********************, паспорт ***********",
        True,
        id="person_and_passport",
    ),
    pytest.param(
        "паспорт 45 09 123456",
        "паспорт ************",
        False,
        id="passport",
    ),
    pytest.param(
        "водительское удостоверение 77 01 123456",
        "водительское удостоверение ************",
        False,
        id="driver_license",
    ),
    pytest.param(
        "Дата рождения: 12.05.1990",
        "Дата рождения: **********",
        False,
        id="date_of_birth_dot",
    ),
    pytest.param(
        "Дата рождения 05/12/1990",
        "Дата рождения **********",
        False,
        id="date_of_birth_slash",
    ),
    pytest.param(
        "Гражданство: РФ",
        "Гражданство: **",
        False,
        id="citizenship",
    ),
    pytest.param(
        "Держатель карты: IVAN IVANOV",
        "Держатель карты: ***********",
        False,
        id="card_holder",
    ),
    pytest.param(
        "родился 12 мая 1990 года в городе Казань",
        "родился **************** в *************",
        True,
        id="birth_date_and_place",
    ),
    pytest.param(
        "выдан ОУФМС России по г. Москве в отделе УФМС",
        "выдан " + "*" * 25 + " в отделе " + "*" * 4,
        True,
        id="passport_issuer",
    ),
    pytest.param(
        "проживает: Россия, 125009, г. Москва, ул. Тверская, д. 15, кв. 45",
        "проживает: " + "*" * 54,
        True,
        id="address",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("phrase", "expected_mask", "natasha_dependent"), MASK_CASES)
async def test_process_mask_retry_unmask(client, phrase, expected_mask, natasha_dependent):
    """Маскирование, ретрай исходника, демаскирование по payload_id."""
    payload_id = f"mask-{phrase[:8]}-{abs(hash(phrase)) % 10000}"

    # 1. Маскирование
    r = await client.post("/process", json={"payload": phrase, "payload_id": payload_id})
    assert r.status_code == 200
    masked = r.json()["result"]

    if natasha_dependent and not _natasha_available():
        # Деградация: Natasha недоступна, сервис не падает.
        if phrase.startswith("Клиент Иванов"):
            # Regex-часть (паспорт) обязана быть верной.
            assert masked.endswith(", паспорт ***********")
        pytest.skip("Natasha unavailable")

    assert masked == expected_mask

    # 2. Повтор исходника возвращает ту же маску
    r2 = await client.post("/process", json={"payload": phrase, "payload_id": payload_id})
    assert r2.status_code == 200
    assert r2.json()["result"] == masked

    # 3. Запрос с маской возвращает исходник
    r3 = await client.post("/process", json={"payload": masked, "payload_id": payload_id})
    assert r3.status_code == 200
    assert r3.json()["result"] == phrase


# Кейсы без маски: фраза должна вернуться без изменений.
NO_MASK_CASES = [
    pytest.param("Поэт Александр Сергеевич Пушкин написал роман", id="historical_poet"),
    pytest.param("Александр Сергеевич Пушкин", id="historical_person"),
    pytest.param(
        "Отделение Альфа-Банка: г. Москва, ул. Каланчевская, д. 27",
        id="bank_branch_address",
    ),
    pytest.param("Московский район отметил юбилей", id="district"),
    pytest.param("NEW YORK is a city", id="latin_city"),
    pytest.param("в 2024 году было 1500 заявок", id="year_and_count"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("phrase",), NO_MASK_CASES)
async def test_process_no_mask(client, phrase):
    """Фразы без ПДн возвращаются без изменений."""
    payload_id = f"nomask-{abs(hash(phrase)) % 10000}"
    r = await client.post("/process", json={"payload": phrase, "payload_id": payload_id})
    assert r.status_code == 200
    assert r.json()["result"] == phrase

    # Повтор исходника возвращает исходник
    r2 = await client.post("/process", json={"payload": phrase, "payload_id": payload_id})
    assert r2.status_code == 200
    assert r2.json()["result"] == phrase


@pytest.mark.asyncio
async def test_process_without_system_id(client):
    """Без X-System-Id запрос проходит (профиль checker)."""
    payload_id = "sys-none-1"
    phrase = "паспорт 4509 123456"
    r = await client.post("/process", json={"payload": phrase, "payload_id": payload_id})
    assert r.status_code == 200
    assert r.json()["result"] == "паспорт ***********"


@pytest.mark.asyncio
async def test_process_unknown_system_403(client):
    """X-System-Id: unknown-system возвращает 403 без исходного текста."""
    payload_id = "sys-unknown-1"
    phrase = "паспорт 4509 123456"
    r = await client.post(
        "/process",
        json={"payload": phrase, "payload_id": payload_id},
        headers={"X-System-Id": "unknown-system"},
    )
    assert r.status_code == 403
    body = r.json()
    assert "error" in body
    assert phrase not in r.text


@pytest.mark.asyncio
async def test_process_disabled_system_403(client):
    """X-System-Id: disabled-system возвращает 403."""
    payload_id = "sys-disabled-1"
    r = await client.post(
        "/process",
        json={"payload": "паспорт 4509 123456", "payload_id": payload_id},
        headers={"X-System-Id": "disabled-system"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_process_demo_no_unmask_masks(client):
    """X-System-Id: demo-no-unmask маскирует, но не демаскирует."""
    payload_id = "sys-nounmask-1"
    phrase = "паспорт 4509 123456"
    r = await client.post(
        "/process",
        json={"payload": phrase, "payload_id": payload_id},
        headers={"X-System-Id": "demo-no-unmask"},
    )
    assert r.status_code == 200
    masked = r.json()["result"]
    assert masked == "паспорт ***********"

    # Повтор маски возвращает маску, не исходник
    r2 = await client.post(
        "/process",
        json={"payload": masked, "payload_id": payload_id},
        headers={"X-System-Id": "demo-no-unmask"},
    )
    assert r2.status_code == 200
    assert r2.json()["result"] == masked
    assert r2.json()["result"] != phrase


@pytest.mark.asyncio
async def test_process_preserves_plain_text(client):
    """Неперсональный текст вне спана сохраняется посимвольно."""
    payload_id = "test-plain-1"
    text = "Обычный текст без ПДн"
    resp = await client.post(
        "/process",
        json={"payload": text, "payload_id": payload_id},
    )
    assert resp.status_code == 200
    assert resp.json()["result"] == text


@pytest.mark.asyncio
async def test_process_rate_limit_retry_after(client, monkeypatch):
    """Перегрузка возвращает 429 с заголовком Retry-After."""
    from app.services.rate_limiter import rate_limiter

    async def deny(*args, **kwargs):
        return False

    monkeypatch.setattr(rate_limiter, "check", deny)
    response = await client.post(
        "/process",
        json={"payload": "тест", "payload_id": "rate-limit-1"},
    )
    assert response.status_code == 429
    assert response.headers.get("retry-after") == "1"


@pytest.mark.asyncio
async def test_process_missing_payload_id(client):
    response = await client.post(
        "/process",
        json={"payload": "Тест"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_process_missing_payload(client):
    response = await client.post(
        "/process",
        json={"payload_id": "some-id"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_process_extra_field_rejected(client):
    response = await client.post(
        "/process",
        json={
            "payload": "Тест",
            "payload_id": "some-id",
            "extra_field": "should_fail",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "redis" in data["checks"]


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "pii_gateway_requests_total" in response.text
