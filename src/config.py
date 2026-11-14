"""Configuración centralizada cargada desde variables de entorno (.env)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from urllib.parse import quote_plus

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"
VALID_THINKING_LEVELS = frozenset({"low", "medium", "high", "xhigh"})
DEPRECATED_MODEL_PREFIXES = ("gemini-2.5-", "gemini-2.0-", "gemini-1.5-", "gemini-1.0-")


def resolve_gemini_model(requested_model: str | None) -> tuple[str, str | None]:
    """Valida el nombre del modelo Gemini y sustituye versiones deprecadas.

    Args:
        requested_model: Nombre de modelo configurado por entorno o interfaz.

    Returns:
        Tupla ``(modelo_efectivo, mensaje_advertencia)``.
    """
    model = (requested_model or "").strip() or DEFAULT_GEMINI_MODEL
    lowered = model.lower()
    if lowered.startswith(DEPRECATED_MODEL_PREFIXES) or lowered == "gemini-pro":
        warning = (
            f"El modelo '{model}' está deprecado en la API actual de Gemini; "
            f"se sustituye automáticamente por '{DEFAULT_GEMINI_MODEL}'."
        )
        logger.warning(warning)
        return DEFAULT_GEMINI_MODEL, warning
    return model, None


def _parse_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Lee un entero desde entorno aplicando límites seguros."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Valor inválido para %s='%s'; usando %d.", name, raw, default)
        return default
    return max(minimum, min(maximum, value))


def build_database_url() -> str:
    """Construye la cadena de conexión PostgreSQL a partir de `.env`."""
    explicit_url = os.getenv("DATABASE_URL", "").strip()
    if explicit_url:
        return explicit_url

    host = os.getenv("POSTGRES_HOST", "localhost").strip() or "localhost"
    port = os.getenv("POSTGRES_PORT", "5432").strip() or "5432"
    db_name = os.getenv("POSTGRES_DB", "transport_rag").strip() or "transport_rag"
    user = quote_plus(os.getenv("POSTGRES_USER", "rag_user").strip() or "rag_user")
    password = quote_plus(os.getenv("POSTGRES_PASSWORD", "rag_password").strip() or "rag_password")
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


@dataclass(frozen=True)
class Settings:
    """Parámetros de configuración de la aplicación RAG."""

    database_url: str
    llm_provider: str
    gemini_api_key: str
    gemini_model: str
    gemini_model_warning: str | None
    gemini_thinking_level: str
    ollama_base_url: str
    ollama_model: str
    gemini_embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    max_pdf_files: int
    default_top_k: int

    @classmethod
    def from_env(cls, *, override: bool = False) -> Settings:
        """Carga las variables de entorno desde `.env` y el sistema."""
        load_dotenv(override=override)

        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provider not in {"gemini", "ollama"}:
            logger.warning("LLM_PROVIDER='%s' no reconocido; usando 'gemini'.", provider)
            provider = "gemini"

        gemini_model, model_warning = resolve_gemini_model(os.getenv("GEMINI_MODEL"))

        thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
        if thinking_level not in VALID_THINKING_LEVELS:
            thinking_level = "low"

        chunk_size = _parse_int("CHUNK_SIZE", default=900, minimum=200, maximum=4000)
        raw_overlap = _parse_int("CHUNK_OVERLAP", default=150, minimum=0, maximum=1000)
        chunk_overlap = min(raw_overlap, chunk_size - 1)

        return cls(
            database_url=build_database_url(),
            llm_provider=provider,
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=gemini_model,
            gemini_model_warning=model_warning,
            gemini_thinking_level=thinking_level,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip() or "llama3.1:8b",
            gemini_embedding_model=os.getenv(
                "GEMINI_EMBEDDING_MODEL",
                "gemini-embedding-001",
            ).strip()
            or "gemini-embedding-001",
            embedding_dim=384,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_pdf_files=_parse_int("MAX_PDF_FILES", default=3, minimum=1, maximum=10),
            default_top_k=_parse_int("DEFAULT_TOP_K", default=5, minimum=3, maximum=8),
        )
