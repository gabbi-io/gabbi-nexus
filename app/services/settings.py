from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "sim", "on"}


@dataclass
class AppSettings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    conversation_window: int = int(os.getenv("CONVERSATION_WINDOW", "6"))

    # Ajustes Nexus/Gabbi
    # Por padrão, CSV local deixa de ser fonte de verdade.
    gabbi_disable_local_knowledge_table: bool = env_bool("GABBI_DISABLE_LOCAL_KNOWLEDGE_TABLE", True)
    gabbi_force_db_tabular: bool = env_bool("GABBI_FORCE_DB_TABULAR", True)
    gabbi_force_rag_first: bool = env_bool("GABBI_FORCE_RAG_FIRST", False)

    # PostgreSQL usado para consultas tabulares vivas.
    # Ordem suportada: GABBI_DATABASE_URL > GABBI_POSTGRES_URL > DATABASE_URL
    gabbi_database_url: str = (
        os.getenv("GABBI_DATABASE_URL", "").strip()
        or os.getenv("GABBI_POSTGRES_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )
    gabbi_tabular_schema: str = os.getenv("GABBI_TABULAR_SCHEMA", "public").strip()
    gabbi_tabular_table: str = os.getenv("GABBI_TABULAR_TABLE", "gabbi_knowledge_table_active").strip()
    gabbi_tabular_max_rows: int = int(os.getenv("GABBI_TABULAR_MAX_ROWS", "200000"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)


settings = AppSettings()
