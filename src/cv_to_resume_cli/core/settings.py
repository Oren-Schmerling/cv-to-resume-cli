"""Phase 2 settings: vLLM server address and local model directory.

Reads .env if present. Later phases add their own settings here.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the matcher (vLLM host/port, model dir)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    hf_model_local_dir: str = "./models/slm"
