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
async def test_process_mask_only(client, sample_text):
    response = await client.post(
        "/process",
        json={
            "system_id": "crm-system",
            "text": sample_text,
            "mode": "mask_only",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "masked_text" in data
    assert "test@mail.ru" not in data["masked_text"]
    assert "[EMAIL_1]" in data["masked_text"]


@pytest.mark.asyncio
async def test_process_detect_only(client, sample_text):
    response = await client.post(
        "/process",
        json={
            "system_id": "crm-system",
            "text": sample_text,
            "mode": "detect_only",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "entities" in data
    assert len(data["entities"]) > 0


@pytest.mark.asyncio
async def test_process_invalid_system(client):
    response = await client.post(
        "/process",
        json={
            "system_id": "unknown-system",
            "text": "Тест",
            "mode": "mask_only",
        },
    )
    # Неизвестная система должна работать с default конфигурацией
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_process_empty_text(client):
    response = await client.post(
        "/process",
        json={
            "system_id": "crm-system",
            "text": "",
            "mode": "mask_only",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_process_missing_system_id(client):
    response = await client.post(
        "/process",
        json={
            "text": "Тест",
            "mode": "mask_only",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_process_extra_field_rejected(client):
    response = await client.post(
        "/process",
        json={
            "system_id": "crm-system",
            "text": "Тест",
            "mode": "mask_only",
            "extra_field": "should_fail",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "pii_gateway_requests_total" in response.text


@pytest.mark.asyncio
async def test_process_full_mode_without_llm(client, sample_text):
    """Full mode без LLM должен вернуть ошибку (нет API key)"""
    response = await client.post(
        "/process",
        json={
            "system_id": "crm-system",
            "text": sample_text,
            "mode": "full",
        },
    )
    # Может быть 400 (pipeline error) или 500 (internal), но не 200
    assert response.status_code in (400, 500)