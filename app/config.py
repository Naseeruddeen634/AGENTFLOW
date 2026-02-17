"""Configuration management for AgentFlow."""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4"))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "2")))
    max_steps: int = field(default_factory=lambda: int(os.getenv("MAX_STEPS", "10")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "2048")))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8001")))

    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")


def get_settings() -> Settings:
    return Settings()
