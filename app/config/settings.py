from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # App
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    reload: bool = False

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    pii_sanitization: bool = True

    # LLM
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout: int = 30
    llm_max_retries: int = 3
    llm_retry_delay: int = 1

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_max_connections: int = 100
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5

    # Rate Limit
    rate_limit_enabled: bool = True
    rate_limit_default_rps: int = 1000
    rate_limit_burst: int = 2000
    rate_limit_key_prefix: str = "pii_gateway:ratelimit:"

    # Payload store
    payload_store_ttl_seconds: int = 3600
    payload_store_key_prefix: str = "pii_gateway:payload:"

    # Detector
    natasha_enabled: bool = True
    natasha_confidence_threshold: float = 0.85
    presidio_enabled: bool = True
    presidio_confidence_threshold: float = 0.8
    presidio_language: str = "ru"
    regex_enabled: bool = True
    composite_strategy: str = "priority"
    composite_deduplicate: bool = True
    composite_min_confidence: float = 0.75

    # Masker
    masker_token_prefix: str = "["
    masker_token_suffix: str = "]"
    masker_token_format: str = "{type}_{index}"
    masker_preserve_case: bool = False
    masker_preserve_length: bool = False

    # Pipeline
    pipeline_max_text_length: int = 100_000
    pipeline_max_entities_per_request: int = 500
    pipeline_llm_timeout: int = 25
    pipeline_total_timeout: int = 28

    # Config paths
    config_dir: Path = Field(default_factory=lambda: Path(__file__).parent)
    pii_rules_path: Path = Field(default_factory=lambda: Path(__file__).parent / "pii_rules.yaml")
    logging_config_path: Path = Field(default_factory=lambda: Path(__file__).parent / "logging.yaml")

    @field_validator("composite_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        if v not in ("priority", "union", "intersection"):
            raise ValueError("composite_strategy must be 'priority', 'union', or 'intersection'")
        return v

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        if v not in ("json", "console"):
            raise ValueError("log_format must be 'json' or 'console'")
        return v


class PIIRulesConfig(BaseSettings):
    """Конфигурация правил ПДн из YAML"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    systems: dict[str, dict[str, Any]] = {}
    default: dict[str, Any] = {}

    @classmethod
    def load_from_yaml(cls, path: Path) -> "PIIRulesConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Разделяем default и systems
        default = data.get("default", {})
        systems = data.get("systems", {})
        return cls(default=default, systems=systems)

    def get_system_config(self, system_id: str) -> dict[str, Any]:
        """Получить конфигурацию для конкретной системы с fallback на default"""
        system_config = self.systems.get(system_id, {})
        # Глубокое слияние с default
        return self._deep_merge(self.default, system_config)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PIIRulesConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


# Глобальные экземпляры настроек
settings = AppSettings()
pii_rules = PIIRulesConfig.load_from_yaml(settings.pii_rules_path)
