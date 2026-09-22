from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class PIIEntityType(str, Enum):
    """Типы персональных данных"""
    PERSON = "PERSON"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    PLACE_OF_BIRTH = "PLACE_OF_BIRTH"
    PASSPORT_SERIES = "PASSPORT_SERIES"
    PASSPORT_NUMBER = "PASSPORT_NUMBER"
    PASSPORT_ISSUER = "PASSPORT_ISSUER"
    PASSPORT_DEPT_CODE = "PASSPORT_DEPT_CODE"
    PASSPORT_ISSUE_DATE = "PASSPORT_ISSUE_DATE"
    CITIZENSHIP = "CITIZENSHIP"
    DRIVER_LICENSE = "DRIVER_LICENSE"
    ADDRESS = "ADDRESS"
    ADDRESS_PARTIAL = "ADDRESS_PARTIAL"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    INN = "INN"
    BANK_CARD = "BANK_CARD"
    CARD_CVV = "CARD_CVV"
    CARD_PIN = "CARD_PIN"
    CARD_HOLDER = "CARD_HOLDER"


class ProcessRequest(BaseModel):
    """Запрос на обработку по контракту AlfaSonar: {payload, payload_id}"""
    model_config = ConfigDict(
        str_strip_whitespace=False,
        validate_assignment=True,
        extra="forbid",
    )

    payload: Annotated[str, StringConstraints(max_length=100_000)] = Field(
        ...,
        description="Строка для обработки. На прямом шаге — исходный текст с ПДн; "
        "на обратном шаге — ранее возвращённая замаскированная строка (тот же payload_id).",
        examples=["Клиент Иванов Иван Иванович, паспорт 4509 123456"],
    )
    payload_id: Annotated[str, StringConstraints(min_length=1, max_length=128)] = Field(
        ...,
        description="Идентификатор корреляции. Один и тот же для пары маскирование→демаскирование.",
        examples=["8a77d363c7c044b49b41d7b8a448243a"],
    )


class ProcessResponse(BaseModel):
    """Ответ по контракту AlfaSonar: {result}"""
    model_config = ConfigDict(frozen=True)

    result: str = Field(..., description="Результат обработки (маска или исходная строка)")


class PIIEntity(BaseModel):
    """Обнаруженная сущность ПДн"""
    model_config = ConfigDict(frozen=True)

    type: PIIEntityType = Field(..., description="Тип ПДн")
    text: str = Field(..., description="Исходный текст сущности")
    start: int = Field(..., ge=0, description="Начальная позиция в тексте")
    end: int = Field(..., ge=0, description="Конечная позиция в тексте")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность детектора")
    detector: str = Field(..., description="Название детектора, нашедшего сущность")
    token: str | None = Field(default=None, description="Маска сущности")


class ErrorResponse(BaseModel):
    """Стандартизированный ответ об ошибке"""
    model_config = ConfigDict(frozen=True)

    error: str = Field(..., description="Код ошибки")
    message: str = Field(..., description="Читаемое описание ошибки")
    details: dict[str, str] | None = Field(default=None, description="Дополнительные детали")
    request_id: str | None = Field(default=None, description="ID запроса для трассировки")


class HealthResponse(BaseModel):
    """Health check ответ"""
    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="Статус сервиса")
    version: str = Field(..., description="Версия сервиса")
    uptime_seconds: float = Field(..., description="Время работы в секундах")
    checks: dict[str, bool] = Field(default_factory=dict, description="Результаты проверок зависимостей")
