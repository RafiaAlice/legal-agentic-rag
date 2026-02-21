from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    """Runtime configuration for the Legal Agentic RAG system.

    Environment variables:
      - GEMINI_API_KEY or GOOGLE_API_KEY: required for Gemini.
      - REBUILD_DB: set to "true" to rebuild the vector DB.
      - DATA_DIR: override data directory.
      - CHROMA_DIR: override chroma persistence directory.
    """

    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data" / "legal_data")))
    CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma_db")))

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "models/gemini-2.0-flash")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    TOP_K: int = int(os.getenv("TOP_K", "5"))

    REBUILD_DB: bool = os.getenv("REBUILD_DB", "false").lower() == "true"

def ensure_dirs(cfg: Config) -> None:
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
