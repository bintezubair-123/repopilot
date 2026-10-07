"""Configuration loaded from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_model: str
    llm_max_retries: int
    embedding_model: str
    chroma_dir: Path
    top_k: int
    log_level: str
    max_diff_chars: int = 24000

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            llm_model=os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash"),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "5")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            chroma_dir=Path(os.getenv("CHROMA_DIR", ".repopilot/chroma")),
            top_k=int(os.getenv("TOP_K", "6")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            max_diff_chars=int(os.getenv("MAX_DIFF_CHARS", "24000")),
        )
