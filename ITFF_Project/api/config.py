from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ITFF_",
        extra="ignore",
    )

    api_token: Optional[str] = None
    api_rate_limit: Optional[str] = "60/minute"
    environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()

