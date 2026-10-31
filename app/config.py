"""Настройки сервиса из переменных окружения."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    model_dir: str = "model"
    cache_dir: str = "/tmp/forecast-cache"
    max_forecasts: int = 20
    max_concurrent: int = 1
    docs: bool = True
    admin_token: str = ""

    @classmethod
    def from_env(cls):
        env = os.environ.get
        return cls(model_dir=env("MODEL_DIR", cls.model_dir), cache_dir=env("CACHE_DIR", cls.cache_dir),
                   max_forecasts=int(env("MAX_FORECASTS", cls.max_forecasts)),
                   max_concurrent=int(env("MAX_CONCURRENT_PREDICTIONS", cls.max_concurrent)),
                   docs=env("DOCS", "1").lower() not in ("0", "false", "no", "off"), admin_token=env("ADMIN_TOKEN", ""))
