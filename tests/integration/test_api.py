from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# Категории ПДн по блоку 4: фраза → точная ожидаемая маска.
# preserved — неперсональный текст, который должен остаться без изменений.
PII_CASES = [
    pytest.param(
        "PERSON",
        "Клиент Иванов Иван Иванович",
        "Клиент И. И. И.",
        ["Клиент "],
        id="person",
    ),
    pytest.param(
        "DATE_OF_BIRTH",
        "Дата рождения: 12.05.1990",
        "Дата рождения: **.**.1990",
        ["Дата рождения: "],
        id="date_of_birth",
    ),
    pytest.param(
        "PLACE_OF_BIRTH",
        "Место рождения: г. Москва",
        "Место рождения: *********",
        ["Место рождения: "],
        id="place_of_birth",
    ),
    pytest.param(
        "PASSPORT",
        "паспорт 4509 123456",
        "паспорт 45******56",
        ["паспорт "],
        id="passport",
    ),
    pytest.param(
        "PASSPORT_SERIES_NUMBER",
        "Серия 4509 номер 123456",
        "Серия 45** номер ****56",
        ["Серия ", " номер "],
        id="passport_series_number",
    ),
    pytest.param(
        "PASSPORT_ISSUER",
        "Орган выдавший паспорт: ОВД района Тверской",
        "Орган выдавший паспорт: *******************",
        ["Орган выдавший паспорт: "],
        id="passport_issuer",
    ),
    pytest.param(
        "PASSPORT_DEPT_CODE",
        "Код подразделения: 770-123",
        "Код подразделения: ***-***",
        ["Код подразделения: "],
        id="passport_dept_code",
    ),
    pytest.param(
        "PASSPORT_ISSUE_DATE",
        "Паспорт выдан 15.03.2015",
        "Паспорт выдан **.**.2015",
        ["Паспорт выдан "],
        id="passport_issue_date",
    ),
    pytest.param(
        "CITIZENSHIP",
        "Гражданство: РФ",
        "Гражданство: **",
        ["Гражданство: "],
        id="citizenship",
    ),
    pytest.param(
        "DRIVER_LICENSE",
        "Водительское удостоверение 77 123456",
        "Водительское удостоверение 77*****56",
        ["Водительское удостоверение "],
        id="driver_license",
    ),
    pytest.param(
        "ADDRESS",
        "Адрес: г. Москва, ул. Тверская, 15, кв. 45",
        "Адрес: ***********************************",
        ["Адрес: "],
        id="address",
    ),
    pytest.param(
        "EMAIL",
        "Email: ivan.ivanov@example.com",
        "Email: i**********@example.com",
        ["Email: "],
        id="email",
    ),
    pytest.param(
        "PHONE",
        "Телефон: +7 (912) 345-67-89",
        "Телефон: +7 (9**) ***-**-89",
        ["Телефон: "],
        id="phone",
    ),
    pytest.param(
        "INN",
        "ИНН: 770123456789",
        "ИНН: 77********89",
        ["ИНН: "],
        id="inn",
    ),
    pytest.param(
        "BANK_CARD",
        "Банковская карта: 4276 1234 5678 9012",
        "Банковская карта: 4276 **** **** 9012",
        ["Банковская карта: "],
        id="bank_card",
    ),
    pytest.param(
        "CARD_CVV",
        "CVV: 123",
        "CVV: ***",
        ["CVV: "],
        id="card_cvv",
    ),
    pytest.param(
        "CARD_PIN",
        "Пин-код: 4321",
        "Пин-код: ****",
        ["Пин-код: "],
        id="card_pin",
    ),
    pytest.param(
        "CARD_HOLDER",
        "Cardholder: IVAN IVANOV",
        "Cardholder: I. I.",
        ["Cardholder: "],
        id="card_holder",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("category", "phrase", "expected_mask", "preserved"), PII_CASES)
async def test_process_category(client, category, phrase, expected_mask, preserved):
    """Для каждой категории: маска, соседний текст, ретрай, демаскирование."""
    payload_id = f"cat-{category}"

    # 1. Маскирование: маска совпала со строкой из фикстуры
    r = await client.post("/process", json={"payload": phrase, "payload_id": payload_id})
    assert r.status_code == 200
    masked = r.json()["result"]
    assert masked == expected_mask

    # 2. Соседний (неперсональный) текст не изменён
    for part in preserved:
        assert part in masked

    # 3. Повтор исходника с тем же payload_id вернул ту же маску
    r2 = await client.post("/process", json={"payload": phrase, "payload_id": payload_id})
    assert r2.status_code == 200
    assert r2.json()["result"] == masked

    # 4. Запрос с маской вернул исходник побайтно
    r3 = await client.post("/process", json={"payload": masked, "payload_id": payload_id})
    assert r3.status_code == 200
    assert r3.json()["result"] == phrase


# Фразы, которые НЕ должны маскироваться (не ПДн).
NO_MASK_CASES = [
    pytest.param("exc-historical", "Александр Сергеевич Пушкин", id="historical_person"),
    pytest.param(
        "exc-bank-branch",
        "Отделение Альфа-Банка находится по адресу: г. Москва, ул. Тверская, 10",
        id="bank_branch_address",
    ),
    pytest.param("exc-year-count", "в 2024 году было 1500 заявок", id="year_and_count"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("payload_id", "phrase"), NO_MASK_CASES)
async def test_process_no_mask(client, payload_id, phrase):
    """Фразы без ПДн возвращаются без изменений."""
    r = await client.post("/process", json={"payload": phrase, "payload_id": payload_id})
    assert r.status_code == 200
    assert r.json()["result"] == phrase


@pytest.mark.asyncio
async def test_process_selfcheck_example(client):
    """Пример из контракта: маскирование и демаскирование по payload_id."""
    payload_id = "selfcheck-1"
    phrase = "Клиент Иванов Иван Иванович, паспорт 4509 123456"
    expected_mask = "Клиент И. И. И., паспорт 45******56"

    r = await client.post("/process", json={"payload": phrase, "payload_id": payload_id})
    assert r.status_code == 200
    assert r.json()["result"] == expected_mask

    r2 = await client.post("/process", json={"payload": expected_mask, "payload_id": payload_id})
    assert r2.status_code == 200
    assert r2.json()["result"] == phrase


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
async def test_process_no_pii_returns_same(client):
    """Без ПДн маска совпадает с исходником, оба шага возвращают ту же строку."""
    payload_id = "test-no-pii-1"
    text = "Обычный текст без персональных данных"
    r1 = await client.post(
        "/process",
        json={"payload": text, "payload_id": payload_id},
    )
    assert r1.status_code == 200
    assert r1.json()["result"] == text

    r2 = await client.post(
        "/process",
        json={"payload": text, "payload_id": payload_id},
    )
    assert r2.status_code == 200
    assert r2.json()["result"] == text


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


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "pii_gateway_requests_total" in response.text
