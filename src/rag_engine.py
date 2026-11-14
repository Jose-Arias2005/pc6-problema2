"""Motor RAG: embeddings con Gemini API, recuperación vectorial y generación con citas."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import requests
from google import genai
from google.genai import types

from src.config import Settings, resolve_gemini_model
from src.database import RetrievedChunk, VectorDatabase
from src.pdf_processor import ProcessedDocument

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "La evidencia disponible en los documentos consultados es insuficiente "
    "para responder esta pregunta."
)

STRICT_SYSTEM_INSTRUCTION = """Eres un asistente técnico especializado en manuales de operación, procedimientos de mantenimiento y protocolos de atención de incidentes de una empresa de transporte.

REGLAS ESTRICTAS DE RESPUESTA Y CONTROL DE EVIDENCIA:
1. Responde SIEMPRE en español claro, técnico y preciso.
2. Usa EXCLUSIVAMENTE la información contenida en los fragmentos de contexto proporcionados en el mensaje del usuario. Está prohibido emplear conocimiento externo, hacer suposiciones o completar procedimientos con datos que no figuren en los fragmentos.
3. Cita obligatoriamente la fuente de cada afirmación relevante usando exactamente el formato:
   [NombreDocumento.pdf, Pág. X]
   respetando el nombre exacto del archivo y el número de página física indicados en cada fragmento.
4. Si la evidencia recuperada no contiene la respuesta o resulta insuficiente para responder con certeza, indícalo explícitamente iniciando con la frase:
   "La evidencia disponible en los documentos consultados es insuficiente para responder esta pregunta."
   Si los fragmentos solo cubren una parte parcial de la consulta, cita exclusivamente lo que sí esté respaldado e indica qué aspecto no aparece en los documentos consultados.
5. Nunca inventes nombres de archivo, páginas, cifras, parámetros, códigos de falla ni pasos operativos."""

EmbeddingProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class RAGResponse:
    """Resultado completo de una consulta RAG con citas y trazabilidad de evidencia."""

    pregunta: str
    respuesta: str
    pasajes: list[RetrievedChunk]
    sin_evidencia: bool
    proveedor_usado: str
    modelo_usado: str
    filtro_documento: str | None
    tiempo_busqueda_ms: float
    tiempo_generacion_ms: float
    advertencia_modelo: str | None = None


def _validate_gemini_api_key(api_key: str) -> str:
    """Valida que exista una clave real de Google Gemini configurada."""
    cleaned = (api_key or "").strip()
    if not cleaned or cleaned == "tu_api_key_de_google_ai_studio":
        raise RuntimeError(
            "Falta configurar GEMINI_API_KEY en el archivo .env para usar los servicios de Gemini."
        )
    return cleaned


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normaliza cada vector fila a norma L2 unitaria para similitud coseno exacta."""
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return (vectors / safe_norms).astype(np.float32)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detecta errores HTTP 429 / RESOURCE_EXHAUSTED de la API de Gemini."""
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "RATE_LIMIT" in msg


class GeminiEmbedder:
    """Generador de embeddings de 384 dimensiones mediante `google-genai`."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-embedding-001",
        *,
        embedding_dim: int = 384,
        batch_size: int = 100,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.model_name = (model_name or "gemini-embedding-001").strip()
        self.embedding_dim = embedding_dim
        self.batch_size = max(1, min(batch_size, 100))
        self.max_retries = max(0, max_retries)
        self.retry_base_delay = max(0.0, retry_base_delay)
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        """Inicializa de forma diferida el cliente oficial `google.genai.Client`."""
        if self._client is None:
            valid_key = _validate_gemini_api_key(self.api_key)
            self._client = genai.Client(api_key=valid_key)
        return self._client

    @property
    def embedding_dimension(self) -> int:
        """Retorna la dimensionalidad configurada para pgvector."""
        return self.embedding_dim

    def _embed_batch_with_retry(
        self,
        contents: list[str],
        *,
        task_type: str,
    ) -> np.ndarray:
        """Llama a `client.models.embed_content` con reintentos exponenciales ante 429."""
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.embedding_dim,
        )
        attempt = 0
        while True:
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                raw_embeddings = getattr(response, "embeddings", None) or []
                if len(raw_embeddings) != len(contents):
                    raise RuntimeError(
                        f"Gemini devolvió {len(raw_embeddings)} embeddings para un lote de "
                        f"{len(contents)} fragmentos."
                    )
                matrix = np.asarray(
                    [item.values for item in raw_embeddings],
                    dtype=np.float32,
                )
                return _l2_normalize(matrix)
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < self.max_retries:
                    delay = self.retry_base_delay * (2**attempt) + random.uniform(0.0, 0.25)
                    logger.warning(
                        "Límite de tasa (429) en embeddings; reintentando en %.2fs (intento %d/%d).",
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        progress_callback: EmbeddingProgressCallback | None = None,
    ) -> np.ndarray:
        """Genera embeddings L2-normalizados por lotes de 50-100 fragmentos."""
        total = len(texts)
        if total == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        effective_batch = max(1, min(batch_size or self.batch_size, 100))
        vectors: list[np.ndarray] = []
        processed = 0

        for start in range(0, total, effective_batch):
            batch = list(texts[start : start + effective_batch])
            vectors.append(
                self._embed_batch_with_retry(batch, task_type="RETRIEVAL_DOCUMENT")
            )
            processed += len(batch)
            if progress_callback is not None:
                progress_callback(processed, total)

        return np.vstack(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        """Convierte una consulta en español en un vector normalizado de 384 dimensiones."""
        cleaned = (query or "").strip()
        if not cleaned:
            raise ValueError("La pregunta no puede estar vacía.")
        matrix = self._embed_batch_with_retry([cleaned], task_type="RETRIEVAL_QUERY")
        return matrix[0]


def build_context_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Construye el prompt de usuario enviando únicamente los fragmentos recuperados."""
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Fragmento {idx}]\n"
            f"Documento: {chunk.nombre_documento}\n"
            f"Página física: {chunk.pagina_fisica}\n"
            f"Contenido:\n{chunk.contenido}"
        )

    evidence_section = "\n\n---\n\n".join(blocks)
    return (
        f"FRAGMENTOS DE EVIDENCIA RECUPERADOS:\n\n"
        f"{evidence_section}\n\n"
        f"PREGUNTA DEL USUARIO:\n{question.strip()}\n\n"
        f"Responde en español citando exactamente como [NombreDocumento.pdf, Pág. X] "
        f"y basándote únicamente en la evidencia anterior."
    )


def _extract_interaction_text(interaction: Any) -> str:
    """Extrae el texto final desde una respuesta de `client.interactions.create`."""
    output_text = getattr(interaction, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: list[str] = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) == "model_output":
            for item in getattr(step, "content", None) or []:
                if getattr(item, "type", None) == "text":
                    text_val = getattr(item, "text", None)
                    if text_val:
                        parts.append(str(text_val))
    return "".join(parts).strip()


def _is_insufficient_evidence_answer(answer: str) -> bool:
    """Detecta si el modelo indicó explícitamente que falta evidencia suficiente."""
    lowered = answer.lower()
    markers = (
        "evidencia disponible en los documentos consultados es insuficiente",
        "evidencia recuperada es insuficiente",
        "no contiene información suficiente",
        "no hay información suficiente",
        "no se encontró evidencia suficiente",
    )
    return any(marker in lowered for marker in markers)


class RAGEngine:
    """Orquesta la ingesta vectorial, búsqueda semántica y generación controlada."""

    def __init__(
        self,
        db: VectorDatabase,
        embedder: GeminiEmbedder,
        settings: Settings,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.settings = settings

    def persist_document(
        self,
        processed_doc: ProcessedDocument,
        *,
        force_replace: bool = False,
        progress_callback: EmbeddingProgressCallback | None = None,
    ) -> ProcessedDocument:
        """Vectoriza (si es necesario) y persiste idempotentemente un PDF procesado."""
        diag = processed_doc.diagnostic
        if diag.tiene_errores_archivo:
            diag.estado_persistencia = "error_archivo"
            return processed_doc

        if not force_replace and self.db.is_document_synced(
            diag.nombre_archivo,
            diag.hash_sha256,
            diag.total_paginas,
            len(processed_doc.chunks),
        ):
            self.db.touch_document_timestamp(diag.nombre_archivo)
            diag.estado_persistencia = "sin_cambios"
            return processed_doc

        texts = [chunk.contenido for chunk in processed_doc.chunks]
        embeddings = self.embedder.embed_texts(
            texts,
            batch_size=100,
            progress_callback=progress_callback,
        )
        action = self.db.upsert_document_idempotent(
            nombre_archivo=diag.nombre_archivo,
            hash_sha256=diag.hash_sha256,
            total_paginas=diag.total_paginas,
            chunks=processed_doc.chunks,
            embeddings=embeddings,
            force_replace=force_replace,
        )
        diag.estado_persistencia = action
        return processed_doc

    def _generate_with_gemini(
        self,
        prompt: str,
        *,
        model_override: str | None = None,
    ) -> tuple[str, str, str | None]:
        """Genera respuesta usando el SDK `google-genai` (Interactions API)."""
        api_key = self.settings.gemini_api_key
        if not api_key or api_key == "tu_api_key_de_google_ai_studio":
            raise RuntimeError(
                "Falta configurar GEMINI_API_KEY en el archivo .env para usar el proveedor Gemini."
            )

        requested_model = model_override or self.settings.gemini_model
        resolved_model, warning = resolve_gemini_model(requested_model)

        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model=resolved_model,
            system_instruction=STRICT_SYSTEM_INSTRUCTION,
            input=prompt,
            generation_config={
                "thinking_level": self.settings.gemini_thinking_level,
                "max_output_tokens": 32768,
            },
            store=False,
        )
        text = _extract_interaction_text(interaction)
        if not text:
            raise RuntimeError("El modelo Gemini devolvió una respuesta vacía.")
        return text, resolved_model, warning

    def _generate_with_ollama(
        self,
        prompt: str,
        *,
        model_override: str | None = None,
    ) -> tuple[str, str]:
        """Genera respuesta usando una instancia local de Ollama."""
        model = (model_override or self.settings.ollama_model).strip()
        endpoint = f"{self.settings.ollama_base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": STRICT_SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=120)
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"No se pudo conectar con Ollama en {self.settings.ollama_base_url}. "
                "Verifica que el servicio Ollama esté ejecutándose (`ollama serve`)."
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError(
                "Tiempo de espera agotado al consultar Ollama local."
            ) from exc

        if response.status_code == 404:
            raise RuntimeError(
                f"El modelo '{model}' no está descargado en Ollama. "
                f"Ejecuta `ollama pull {model}` e inténtalo nuevamente."
            )
        response.raise_for_status()

        data = response.json()
        content = str(data.get("message", {}).get("content", "")).strip()
        if not content:
            raise RuntimeError("Ollama devolvió una respuesta vacía.")
        return content, model

    def answer_question(
        self,
        question: str,
        *,
        top_k: int = 5,
        documento_nombre: str | None = None,
        provider_override: str | None = None,
        model_override: str | None = None,
    ) -> RAGResponse:
        """Ejecuta recuperación semántica en `pgvector` y generación con citas estrictas."""
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            raise ValueError("Escribe una pregunta antes de consultar los manuales.")

        clamped_k = max(3, min(int(top_k), 8))
        provider = (provider_override or self.settings.llm_provider).strip().lower()
        if provider not in {"gemini", "ollama"}:
            raise ValueError(f"Proveedor LLM no soportado: '{provider}'. Use 'gemini' u 'ollama'.")

        active_filter = (
            None
            if not documento_nombre or documento_nombre == "Todos los documentos"
            else documento_nombre
        )

        search_start = time.perf_counter()
        query_vector = self.embedder.embed_query(cleaned_question)
        passages = self.db.search_similar_chunks(
            query_vector,
            top_k=clamped_k,
            documento_nombre=active_filter,
        )
        search_ms = (time.perf_counter() - search_start) * 1000.0

        if not passages:
            default_model = (
                self.settings.gemini_model if provider == "gemini" else self.settings.ollama_model
            )
            return RAGResponse(
                pregunta=cleaned_question,
                respuesta=INSUFFICIENT_EVIDENCE_MESSAGE,
                pasajes=[],
                sin_evidencia=True,
                proveedor_usado=provider,
                modelo_usado=model_override or default_model,
                filtro_documento=active_filter,
                tiempo_busqueda_ms=search_ms,
                tiempo_generacion_ms=0.0,
            )

        prompt = build_context_prompt(cleaned_question, passages)
        gen_start = time.perf_counter()
        model_warning: str | None = None

        if provider == "gemini":
            answer, used_model, model_warning = self._generate_with_gemini(
                prompt,
                model_override=model_override,
            )
        else:
            answer, used_model = self._generate_with_ollama(
                prompt,
                model_override=model_override,
            )

        gen_ms = (time.perf_counter() - gen_start) * 1000.0

        return RAGResponse(
            pregunta=cleaned_question,
            respuesta=answer,
            pasajes=passages,
            sin_evidencia=_is_insufficient_evidence_answer(answer),
            proveedor_usado=provider,
            modelo_usado=used_model,
            filtro_documento=active_filter,
            tiempo_busqueda_ms=search_ms,
            tiempo_generacion_ms=gen_ms,
            advertencia_modelo=model_warning or self.settings.gemini_model_warning,
        )
