"""Pruebas unitarias para el motor RAG, citas estrictas y proveedores Gemini/Ollama."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

from src.config import Settings, resolve_gemini_model
from src.database import RetrievedChunk
from src.rag_engine import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    STRICT_SYSTEM_INSTRUCTION,
    GeminiEmbedder,
    RAGEngine,
    build_context_prompt,
)


def _make_settings(*, provider: str = "gemini", api_key: str = "fake-key") -> Settings:
    return Settings(
        database_url="postgresql://rag:rag@localhost:5432/rag",
        llm_provider=provider,
        gemini_api_key=api_key,
        gemini_model="gemini-4-hydrogen",
        gemini_model_warning=None,
        gemini_thinking_level="low",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1:8b",
        gemini_embedding_model="gemini-embedding-001",
        embedding_dim=384,
        chunk_size=900,
        chunk_overlap=150,
        max_pdf_files=3,
        default_top_k=5,
    )


def _sample_passages() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            fragmento_id=10,
            nombre_documento="Protocolo_Incidentes.pdf",
            pagina_fisica=14,
            indice_fragmento=22,
            contenido="Ante pérdida de presión neumática, detenga el autobús en zona segura y active luces de emergencia.",
            similitud=0.89,
            distancia_coseno=0.11,
        )
    ]


def test_resolve_gemini_model_substitutes_deprecated_models() -> None:
    model, warning = resolve_gemini_model("gemini-2.5-flash")
    assert model == "gemini-4-hydrogen"
    assert warning is not None
    assert "gemini-2.5-flash" in warning

    ok_model, ok_warning = resolve_gemini_model("gemini-4-hydrogen")
    assert ok_model == "gemini-4-hydrogen"
    assert ok_warning is None


def test_build_context_prompt_contains_only_retrieved_evidence_and_metadata() -> None:
    passages = _sample_passages()
    prompt = build_context_prompt("¿Qué hacer si falla la presión neumática?", passages)
    assert "Documento: Protocolo_Incidentes.pdf" in prompt
    assert "Página física: 14" in prompt
    assert "luces de emergencia" in prompt
    assert "[NombreDocumento.pdf, Pág. X]" in prompt


def test_answer_question_returns_insufficient_when_no_chunks_found() -> None:
    db = MagicMock()
    db.search_similar_chunks.return_value = []
    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(384, dtype=np.float32)

    engine = RAGEngine(db=db, embedder=embedder, settings=_make_settings())
    response = engine.answer_question("¿Cuál es el horario del comedor?", top_k=5)

    assert response.sin_evidencia is True
    assert response.respuesta == INSUFFICIENT_EVIDENCE_MESSAGE
    assert response.pasajes == []


def test_answer_question_with_gemini_interactions_api(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_calls: list[dict] = []

    class _FakeInteractions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            captured_calls.append(kwargs)
            return SimpleNamespace(
                output_text=(
                    "Debe detener el autobús en zona segura y activar luces de emergencia "
                    "[Protocolo_Incidentes.pdf, Pág. 14]."
                )
            )

    class _FakeGenAIClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.interactions = _FakeInteractions()

    monkeypatch.setattr("src.rag_engine.genai.Client", _FakeGenAIClient)

    db = MagicMock()
    db.search_similar_chunks.return_value = _sample_passages()
    embedder = MagicMock()
    embedder.embed_query.return_value = np.ones(384, dtype=np.float32)

    engine = RAGEngine(db=db, embedder=embedder, settings=_make_settings(provider="gemini"))
    response = engine.answer_question(
        "¿Qué hacer ante pérdida de presión neumática?",
        top_k=5,
        documento_nombre="Todos los documentos",
    )

    assert response.sin_evidencia is False
    assert "[Protocolo_Incidentes.pdf, Pág. 14]" in response.respuesta
    assert response.proveedor_usado == "gemini"
    assert response.modelo_usado == "gemini-4-hydrogen"
    assert response.filtro_documento is None
    assert len(captured_calls) == 1

    call = captured_calls[0]
    assert call["model"] == "gemini-4-hydrogen"
    assert call["system_instruction"] == STRICT_SYSTEM_INSTRUCTION
    assert "temperature" not in call["generation_config"]
    assert call["generation_config"]["thinking_level"] == "low"
    assert call["store"] is False


def test_answer_question_with_ollama_and_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = MagicMock()
    db.search_similar_chunks.return_value = _sample_passages()
    embedder = MagicMock()
    embedder.embed_query.return_value = np.ones(384, dtype=np.float32)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "message": {
            "content": "La evidencia disponible en los documentos consultados es insuficiente para responder esta pregunta."
        }
    }
    monkeypatch.setattr("src.rag_engine.requests.post", MagicMock(return_value=fake_resp))

    engine = RAGEngine(db=db, embedder=embedder, settings=_make_settings(provider="ollama"))
    response = engine.answer_question("Pregunta fuera de alcance", top_k=3)

    assert response.sin_evidencia is True
    assert response.proveedor_usado == "ollama"

    monkeypatch.setattr(
        "src.rag_engine.requests.post",
        MagicMock(side_effect=requests.ConnectionError("refused")),
    )
    with pytest.raises(RuntimeError, match="Ollama"):
        engine.answer_question("¿Qué hacer?", top_k=3)


def test_gemini_embedder_batches_normalizes_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class _FakeModels:
        def embed_content(self, *, model, contents, config):  # type: ignore[no-untyped-def]
            calls.append({"model": model, "contents": list(contents), "config": config})
            embeddings = []
            for idx, _ in enumerate(contents):
                vec = [0.0] * 384
                vec[idx] = 3.0
                vec[idx + 1] = 4.0  # norma = 5.0 -> debe normalizarse a 1.0
                embeddings.append(SimpleNamespace(values=vec))
            return SimpleNamespace(embeddings=embeddings)

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = _FakeModels()

    monkeypatch.setattr("src.rag_engine.genai.Client", _FakeClient)

    embedder = GeminiEmbedder(
        api_key="fake-api-key",
        model_name="gemini-embedding-001",
        embedding_dim=384,
        batch_size=2,
    )
    progress_events: list[tuple[int, int]] = []
    result = embedder.embed_texts(
        ["frag 1", "frag 2", "frag 3", "frag 4", "frag 5"],
        progress_callback=lambda done, total: progress_events.append((done, total)),
    )

    assert len(calls) == 3
    assert all(c["model"] == "gemini-embedding-001" for c in calls)
    assert all(c["config"].output_dimensionality == 384 for c in calls)
    assert all(c["config"].task_type == "RETRIEVAL_DOCUMENT" for c in calls)
    assert result.shape == (5, 384)
    assert result.dtype == np.float32
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-5)
    assert progress_events == [(2, 5), (4, 5), (5, 5)]


def test_gemini_embedder_retries_on_429_and_embeds_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}
    captured_task_types: list[str | None] = []

    class _FakeModels:
        def embed_content(self, *, model, contents, config):  # type: ignore[no-untyped-def]
            attempts["count"] += 1
            captured_task_types.append(config.task_type)
            if attempts["count"] == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
            vec = [2.0] + [0.0] * 383
            return SimpleNamespace(embeddings=[SimpleNamespace(values=vec)])

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = _FakeModels()

    monkeypatch.setattr("src.rag_engine.genai.Client", _FakeClient)
    monkeypatch.setattr("src.rag_engine.time.sleep", lambda _: None)

    embedder = GeminiEmbedder(
        api_key="fake-api-key",
        retry_base_delay=0.01,
        max_retries=3,
    )
    query_vec = embedder.embed_query("¿Qué hacer ante falla de frenos?")

    assert attempts["count"] == 2
    assert captured_task_types == ["RETRIEVAL_QUERY", "RETRIEVAL_QUERY"]
    assert query_vec.shape == (384,)
    assert np.linalg.norm(query_vec) == pytest.approx(1.0)


def test_gemini_embedder_requires_valid_api_key() -> None:
    embedder = GeminiEmbedder(api_key="tu_api_key_de_google_ai_studio")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        embedder.embed_query("consulta")
