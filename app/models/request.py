from enum import Enum
from typing import Annotated
from pydantic import BaseModel, Field, StringConstraints, ConfigDict


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


class ProcessingMode(str, Enum):
    """Режим обработки"""
    FULL = "full"           # detect -> mask -> LLM -> unmask
    MASK_ONLY = "mask_only" # только маскирование
    UNMASK_ONLY = "unmask_only" # только демаскирование
    DETECT_ONLY = "detect_only" # только детекция


class ProcessRequest(BaseModel):
    """Запрос на обработку текста"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    system_id: Annotated[str, StringConstraints(min_length=1, max_length=64)] = Field(
        ...,
        description="Идентификатор системы-потребителя",
        examples=["crm-system", "loan-scoring", "support-chat"],
    )
    text: Annotated[str, StringConstraints(min_length=1, max_length=100_000)] = Field(
        ...,
        description="Текст для обработки",
        examples=["Уважаемый Иван Иванович Иванов, ваш паспорт 4500 123456..."],
    )
    mode: ProcessingMode = Field(
        default=ProcessingMode.FULL,
        description="Режим обработки",
    )
    llm_prompt: str | None = Field(
        default=None,
        description="Дополнительный промпт для LLM (опционально)",
        max_length=5000,
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Произвольные метаданные запроса",
    )


class PIIEntity(BaseModel):
    """Обнаруженная сущность ПДн"""
    model_config = ConfigDict(frozen=True)

    type: PIIEntityType = Field(..., description="Тип ПДн")
    text: str = Field(..., description="Исходный текст сущности")
    start: int = Field(..., ge=0, description="Начальная позиция в тексте")
    end: int = Field(..., ge=0, description="Конечная позиция в тексте")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность детектора")
    detector: str = Field(..., description="Название детектора, нашедшего сущность")
    token: str | None = Field(default=None, description="Токен маскирования (после маскирования)")


class DetectOnlyResponse(BaseModel):
    """Ответ при режиме detect_only"""
    model_config = ConfigDict(frozen=True)

    entities: list[PIIEntity] = Field(default_factory=list, description="Найденные сущности ПДн")
    processing_time_ms: float = Field(..., description="Время обработки в мс")


class MaskOnlyResponse(BaseModel):
    """Ответ при режиме mask_only"""
    model_config = ConfigDict(frozen=True)

    masked_text: str = Field(..., description="Замаскированный текст")
    entities: list[PIIEntity] = Field(default_factory=list, description="Найденные сущности ПДн")
    processing_time_ms: float = Field(..., description="Время обработки в мс")


class ProcessResponse(BaseModel):
    """Полный ответ обработки (full mode)"""
    model_config = ConfigDict(frozen=True)

    original_text: str = Field(..., description="Исходный текст (для проверки)")
    masked_text: str = Field(..., description="Текст после маскирования (отправлен в LLM)")
    llm_response: str = Field(..., description="Ответ LLM (еще замаскированный)")
    unmasked_response: str = Field(..., description="Финальный ответ после демаскирования")
    entities: list[PIIEntity] = Field(default_factory=list, description="Все обнаруженные сущности")
    processing_time_ms: float = Field(..., description="Общее время обработки в мс")
    llm_time_ms: float = Field(..., description="Время ответа LLM в мс")


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