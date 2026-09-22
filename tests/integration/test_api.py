from __future__ import annotations
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_process_mask_then_unmask(client, sample_text):
    """Прямая проверка: маскирование, затем демаскирование по тому же payload_id."""
    payload_id = "test-mask-unmask-1"

    # 1. Маскирование
    mask_resp = await client.post(
        "/process",
        json={"payload": sample_text, "payload_id": payload_id},
    )
    assert mask_resp.status_code == 200
    masked = mask_resp.json()["result"]
    # ПДн не должны остаться в открытом виде
    assert "test@mail.ru" not in masked
    assert "770123456789" not in masked
    assert "4276 1234 5678 9012" not in masked

    # 2. Демаскирование
    unmask_resp = await client.post(
        "/process",
        json={"payload": masked, "payload_id": payload_id},
    )
    assert unmask_resp.status_code == 200
    assert unmask_resp.json()["result"] == sample_text


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
async def test_process_partial_mask_format(client):
    """Маска ФИО и паспорта в стиле инициалов/частичной маски."""
    payload_id = "test-format-1"
    text = "Клиент Иванов Иван Иванович, паспорт 4509 123456"
    resp = await client.post(
        "/process",
        json={"payload": text, "payload_id": payload_id},
    )
    assert resp.status_code == 200
    masked = resp.json()["result"]
    # ФИО → инициалы
    assert "Иванов Иван Иванович" not in masked
    assert "И. И. И." in masked
    # Паспорт → частичная маска
    assert "4509 123456" not in masked
    assert "45" in masked and "56" in masked


@pytest.mark.asyncio
async def test_process_mask_retry(client, sample_text):
    """Ретрай маскирования с тем же исходником возвращает ту же маску."""
    payload_id = "test-retry-1"
    r1 = await client.post(
        "/process",
        json={"payload": sample_text, "payload_id": payload_id},
    )
    assert r1.status_code == 200
    masked1 = r1.json()["result"]

    r2 = await client.post(
        "/process",
        json={"payload": sample_text, "payload_id": payload_id},
    )
    assert r2.status_code == 200
    assert r2.json()["result"] == masked1


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
async def test_metrics_endpoint(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "pii_gateway_requests_total" in response.text