"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # MongoDB
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "geometra"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # File Storage
    upload_dir: Path = Path("./uploads")
    output_dir: Path = Path("./outputs")

    # Agent Constraints
    max_file_size_mb: int = 50
    supported_2d_formats: list[str] = ["pdf", "png", "jpeg", "tiff", "dxf", "dwg"]
    supported_3d_formats: list[str] = ["step", "stp", "iges", "stl", "obj"]

    # Celery
    celery_task_always_eager: bool = False

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
