"""Pruebas unitarias para la carga y validación de configuración."""

from __future__ import annotations

import pytest

from src.config import Settings, build_database_url, resolve_gemini_model


def test_build_database_url_encodes_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_HOST", "db.local")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "manuales")
    monkeypatch.setenv("POSTGRES_USER", "usr@corp")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss:w/rd")

    url = build_database_url()
    assert url == "postgresql://usr%40corp:p%40ss%3Aw%2Frd@db.local:5433/manuales"


def test_build_database_url_prefers_explicit_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://custom:secret@host:5432/mydb")
    assert build_database_url() == "postgresql://custom:secret@host:5432/mydb"


def test_settings_from_env_clamps_values_and_substitutes_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "invalid_provider")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "MEDIUM")
    monkeypatch.setenv("CHUNK_SIZE", "900")
    monkeypatch.setenv("CHUNK_OVERLAP", "950")
    monkeypatch.setenv("DEFAULT_TOP_K", "99")

    settings = Settings.from_env()
    assert settings.llm_provider == "gemini"
    assert settings.gemini_model == "gemini-4-hydrogen"
    assert settings.gemini_model_warning is not None
    assert settings.gemini_thinking_level == "medium"
    assert settings.chunk_overlap == 899
    assert settings.default_top_k == 8
    assert settings.gemini_embedding_model == "gemini-embedding-001"


def test_resolve_gemini_model_preserves_current_models() -> None:
    for model_id in ("gemini-4-hydrogen", "gemini-4-argon", "gemini-3.5-flash-lite"):
        resolved, warning = resolve_gemini_model(model_id)
        assert resolved == model_id
        assert warning is None
