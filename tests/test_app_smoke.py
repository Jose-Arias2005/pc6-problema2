"""Pruebas de humo de la interfaz Streamlit mediante `streamlit.testing.v1.AppTest`."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from streamlit.testing.v1 import AppTest

from src.database import DocumentRecord, InventoryStats, RetrievedChunk

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_renders_gracefully_when_postgres_is_offline(monkeypatch) -> None:
    """Verifica que la app cargue pestañas y barra lateral sin caer si PostgreSQL no responde."""
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "59999")

    at = AppTest.from_file(str(APP_PATH), default_timeout=15)
    at.run()

    assert not at.exception
    assert len(at.tabs) == 3
    assert len(at.sidebar.metric) >= 2
    assert len(at.sidebar.error) >= 1


def test_app_connected_query_renders_answer_and_passages(monkeypatch) -> None:
    """Verifica el flujo completo conectado: inventario en sidebar, consulta y pasajes."""
    fake_docs = [
        DocumentRecord(
            id=1,
            nombre_archivo="Manual_Operaciones.pdf",
            hash_sha256="a" * 64,
            total_paginas=120,
            total_fragmentos=240,
            fecha_ingesta=datetime.now(timezone.utc),
        )
    ]
    fake_stats = InventoryStats(
        total_documentos=1,
        total_fragmentos=240,
        total_paginas=120,
    )
    fake_passages = [
        RetrievedChunk(
            fragmento_id=7,
            nombre_documento="Manual_Operaciones.pdf",
            pagina_fisica=18,
            indice_fragmento=34,
            contenido="Verificar presión neumática mínima de 8 bar antes de iniciar la marcha.",
            similitud=0.9123,
            distancia_coseno=0.0877,
        )
    ]

    class _FakeDB:
        def __init__(self, dsn: str, embedding_dim: int = 384) -> None:
            self.dsn = dsn

        def initialize_schema(self) -> None:
            return None

        def health_check(self) -> tuple[bool, str]:
            return True, "PostgreSQL conectado · pgvector v0.8.0"

        def get_inventory_stats(self) -> InventoryStats:
            return fake_stats

        def list_documents(self) -> list[DocumentRecord]:
            return fake_docs

        def search_similar_chunks(self, query_embedding, *, top_k=5, documento_nombre=None):
            return fake_passages

        def get_sample_chunks(self, nombre_archivo: str, limit: int = 5) -> list[dict]:
            return [{"pagina_fisica": 18, "indice_fragmento": 34, "contenido": fake_passages[0].contenido}]

    class _FakeEmbedder:
        def __init__(self, *args, **kwargs) -> None:
            self.embedding_dim = 384

        def embed_query(self, query: str) -> np.ndarray:
            return np.ones(384, dtype=np.float32)

    class _FakeInteractions:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text="La presión mínima requerida es 8 bar [Manual_Operaciones.pdf, Pág. 18]."
            )

    class _FakeGenAIClient:
        def __init__(self, api_key: str) -> None:
            self.interactions = _FakeInteractions()

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr("src.database.VectorDatabase", _FakeDB)
    monkeypatch.setattr("src.rag_engine.GeminiEmbedder", _FakeEmbedder)
    monkeypatch.setattr("src.rag_engine.genai.Client", _FakeGenAIClient)

    at = AppTest.from_file(str(APP_PATH), default_timeout=15)
    at.run()

    assert not at.exception
    assert at.sidebar.metric[0].value == "1"
    assert at.sidebar.metric[1].value == "240"

    at.text_area(key="question_draft").set_value("¿Cuál es la presión mínima neumática?")
    at.button[3].click().run()  # Botón submit del formulario tras los 3 botones de ejemplo

    assert not at.exception
    rendered_markdown = "\n".join(m.value for m in at.markdown)
    assert "[Manual_Operaciones.pdf, Pág. 18]" in rendered_markdown
    rendered_text = "\n".join(t.value for t in at.text)
    assert "8 bar" in rendered_text
