from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Добавляем корень проекта в path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_text() -> str:
    """Текст с различными типами ПДн"""
    return (
        "Уважаемый Иван Иванович Иванов!\n"
        "Ваш паспорт: 4500 123456, выдан 15.03.2015\n"
        "Код подразделения: 770-123\n"
        "Дата рождения: 12.05.1990\n"
        "Место рождения: г. Москва\n"
        "Гражданство: РФ\n"
        "Адрес: г. Москва, ул. Тверская, 15, кв. 45\n"
        "Email: ivan.ivanov@example.com\n"
        "Телефон: +7 (912) 345-67-89\n"
        "ИНН: 770123456789\n"
        "Банковская карта: 4276 1234 5678 9012\n"
        "CVV: 123\n"
        "Пин-код: 4321\n"
    )
