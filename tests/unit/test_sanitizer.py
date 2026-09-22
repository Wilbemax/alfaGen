from __future__ import annotations
import logging
import pytest

from app.utils.sanitizer import PIISanitizingFilter, sanitize_text


def test_sanitize_email():
    result = sanitize_text("Email: test@mail.ru")
    assert "test@mail.ru" not in result
    assert "[REDACTED]" in result


def test_sanitize_phone():
    result = sanitize_text("Телефон: +7 999 123-45-67")
    assert "+7 999 123-45-67" not in result
    assert "[REDACTED]" in result


def test_sanitize_card():
    result = sanitize_text("Карта: 4276 1234 5678 9012")
    assert "4276 1234 5678 9012" not in result
    assert "[REDACTED]" in result


def test_sanitize_inn():
    result = sanitize_text("ИНН: 770123456789")
    assert "770123456789" not in result
    assert "[REDACTED]" in result


def test_sanitize_passport():
    result = sanitize_text("Паспорт: 4500 123456")
    assert "4500 123456" not in result
    assert "[REDACTED]" in result


def test_sanitize_date():
    result = sanitize_text("Дата рождения: 12.05.1990")
    assert "12.05.1990" not in result
    assert "[REDACTED]" in result


def test_sanitize_plain_text_unchanged():
    text = "Обычный текст без ПДн"
    assert sanitize_text(text) == text


def test_sanitizing_filter():
    filter_ = PIISanitizingFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Email: test@mail.ru, телефон: +7 999 123-45-67",
        args=(),
        exc_info=None,
    )
    assert filter_.filter(record)
    assert "test@mail.ru" not in record.msg
    assert "+7 999 123-45-67" not in record.msg
    assert "[REDACTED]" in record.msg


def test_sanitizing_filter_with_args():
    filter_ = PIISanitizingFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="User %s logged in",
        args=("test@mail.ru",),
        exc_info=None,
    )
    assert filter_.filter(record)
    assert "test@mail.ru" not in str(record.args)
    assert "[REDACTED]" in str(record.args)