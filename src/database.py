"""Capa de persistencia PostgreSQL + pgvector con ingesta idempotente y búsqueda HNSW."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from src.pdf_processor import TextChunk

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
INSERT_BATCH_SIZE = 500


@dataclass(frozen=True)
class DocumentRecord:
    """Metadatos de un documento almacenado en PostgreSQL."""

    id: int
    nombre_archivo: str
    hash_sha256: str
    total_paginas: int
    total_fragmentos: int
    fecha_ingesta: datetime


@dataclass(frozen=True)
class InventoryStats:
    """Contadores globales en tiempo real desde PostgreSQL."""

    total_documentos: int
    total_fragmentos: int
    total_paginas: int


@dataclass(frozen=True)
class RetrievedChunk:
    """Fragmento recuperado por similitud semántica junto con su puntaje coseno."""

    fragmento_id: int
    nombre_documento: str
    pagina_fisica: int
    indice_fragmento: int
    contenido: str
    similitud: float
    distancia_coseno: float


class VectorDatabase:
    """Administra el esquema `pgvector`, la persistencia idempotente y la búsqueda coseno."""

    def __init__(self, dsn: str, embedding_dim: int = EMBEDDING_DIM) -> None:
        if not dsn or not dsn.strip():
            raise ValueError("La cadena de conexión PostgreSQL (DSN) no puede estar vacía.")
        self.dsn = dsn
        self.embedding_dim = embedding_dim

    @contextmanager
    def _connect(self, *, register_pgvector: bool = True) -> Iterator[psycopg.Connection]:
        """Abre una conexión PostgreSQL y registra el adaptador de `pgvector`."""
        conn = psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5)
        try:
            if register_pgvector:
                register_vector(conn)
            yield conn
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Crea la extensión `vector`, tablas relacionales e índice HNSW si no existen."""
        # Primero se crea la extensión sin register_vector para evitar errores en BD vacía.
        with self._connect(register_pgvector=False) as conn:
            with conn.transaction():
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        with self._connect(register_pgvector=True) as conn:
            with conn.transaction():
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documentos (
                        id SERIAL PRIMARY KEY,
                        nombre_archivo TEXT NOT NULL UNIQUE,
                        hash_sha256 VARCHAR(64) NOT NULL,
                        total_paginas INTEGER NOT NULL CHECK (total_paginas >= 0),
                        fecha_ingesta TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS fragmentos (
                        id SERIAL PRIMARY KEY,
                        documento_id INTEGER NOT NULL
                            REFERENCES documentos(id) ON DELETE CASCADE,
                        pagina_fisica INTEGER NOT NULL CHECK (pagina_fisica >= 1),
                        indice_fragmento INTEGER NOT NULL CHECK (indice_fragmento >= 0),
                        contenido TEXT NOT NULL,
                        embedding vector({self.embedding_dim}) NOT NULL,
                        CONSTRAINT uq_fragmento_documento_indice
                            UNIQUE (documento_id, indice_fragmento)
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_fragmentos_documento_id
                        ON fragmentos (documento_id);
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_fragmentos_embedding_hnsw
                        ON fragmentos
                        USING hnsw (embedding vector_cosine_ops);
                    """
                )

    def health_check(self) -> tuple[bool, str]:
        """Verifica conectividad con PostgreSQL y disponibilidad de la extensión `vector`."""
        try:
            with self._connect(register_pgvector=False) as conn:
                row = conn.execute(
                    "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
                ).fetchone()
            if not row:
                return False, "Conectado a PostgreSQL, pero la extensión 'vector' aún no está activa."
            return True, f"PostgreSQL conectado · pgvector v{row['extversion']}"
        except Exception as exc:
            return False, f"Sin conexión a PostgreSQL: {exc}"

    def is_document_synced(
        self,
        nombre_archivo: str,
        hash_sha256: str,
        total_paginas: int,
        expected_fragment_count: int,
    ) -> bool:
        """Comprueba si el documento ya está persistido con idéntico hash y conteo."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    d.hash_sha256,
                    d.total_paginas,
                    COUNT(f.id) AS fragment_count
                FROM documentos d
                LEFT JOIN fragmentos f ON f.documento_id = d.id
                WHERE d.nombre_archivo = %s
                GROUP BY d.id;
                """,
                (nombre_archivo,),
            ).fetchone()

        if not row:
            return False
        return (
            row["hash_sha256"] == hash_sha256
            and int(row["total_paginas"]) == total_paginas
            and int(row["fragment_count"]) == expected_fragment_count
        )

    def find_duplicate_hash_document(
        self,
        hash_sha256: str,
        exclude_filename: str | None = None,
    ) -> str | None:
        """Devuelve el nombre de otro archivo con el mismo SHA-256 si existe."""
        if not hash_sha256:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT nombre_archivo
                FROM documentos
                WHERE hash_sha256 = %s
                  AND (%s::text IS NULL OR nombre_archivo <> %s::text)
                LIMIT 1;
                """,
                (hash_sha256, exclude_filename, exclude_filename),
            ).fetchone()
        return row["nombre_archivo"] if row else None

    def touch_document_timestamp(self, nombre_archivo: str) -> None:
        """Actualiza `fecha_ingesta` de un documento idéntico verificado en transacción."""
        with self._connect() as conn:
            with conn.transaction():
                conn.execute(
                    "UPDATE documentos SET fecha_ingesta = NOW() WHERE nombre_archivo = %s;",
                    (nombre_archivo,),
                )

    def _validate_embeddings(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Valida longitud y dimensionalidad exacta de los vectores antes de persistir."""
        if len(chunks) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise ValueError(
                f"Desajuste entre fragmentos ({len(chunks)}) y embeddings ({matrix.shape})."
            )
        if matrix.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Dimensión de embedding inválida: esperada {self.embedding_dim}, "
                f"recibida {matrix.shape[1]}."
            )
        return matrix

    def _insert_chunks_in_batches(
        self,
        conn: psycopg.Connection,
        documento_id: int,
        chunks: Sequence[TextChunk],
        embedding_matrix: np.ndarray,
    ) -> None:
        """Inserta fragmentos y vectores por lotes dentro de la transacción activa."""
        if not chunks:
            return

        insert_sql = """
            INSERT INTO fragmentos (
                documento_id,
                pagina_fisica,
                indice_fragmento,
                contenido,
                embedding
            )
            VALUES (%s, %s, %s, %s, %s);
        """
        with conn.cursor() as cur:
            for start in range(0, len(chunks), INSERT_BATCH_SIZE):
                end = start + INSERT_BATCH_SIZE
                batch_rows = [
                    (
                        documento_id,
                        chunk.pagina_fisica,
                        chunk.indice_fragmento,
                        chunk.contenido,
                        embedding_matrix[idx],
                    )
                    for idx, chunk in enumerate(chunks[start:end], start=start)
                ]
                cur.executemany(insert_sql, batch_rows)

    def upsert_document_idempotent(
        self,
        *,
        nombre_archivo: str,
        hash_sha256: str,
        total_paginas: int,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]] | np.ndarray,
        force_replace: bool = False,
    ) -> str:
        """Inserta o actualiza atómicamente un documento y sus fragmentos sin duplicar registros.

        Returns:
            ``"insertado"``, ``"actualizado"`` o ``"sin_cambios"``.
        """
        embedding_matrix = self._validate_embeddings(chunks, embeddings)

        with self._connect() as conn:
            with conn.transaction():
                existing = conn.execute(
                    """
                    SELECT id, hash_sha256, total_paginas
                    FROM documentos
                    WHERE nombre_archivo = %s
                    FOR UPDATE;
                    """,
                    (nombre_archivo,),
                ).fetchone()

                if existing is not None:
                    doc_id = int(existing["id"])
                    count_row = conn.execute(
                        "SELECT COUNT(*) AS cnt FROM fragmentos WHERE documento_id = %s;",
                        (doc_id,),
                    ).fetchone()
                    existing_chunk_count = int(count_row["cnt"]) if count_row else 0

                    is_identical = (
                        existing["hash_sha256"] == hash_sha256
                        and int(existing["total_paginas"]) == total_paginas
                        and existing_chunk_count == len(chunks)
                    )
                    if is_identical and not force_replace:
                        conn.execute(
                            "UPDATE documentos SET fecha_ingesta = NOW() WHERE id = %s;",
                            (doc_id,),
                        )
                        return "sin_cambios"

                    conn.execute(
                        """
                        UPDATE documentos
                        SET hash_sha256 = %s,
                            total_paginas = %s,
                            fecha_ingesta = NOW()
                        WHERE id = %s;
                        """,
                        (hash_sha256, total_paginas, doc_id),
                    )
                    conn.execute("DELETE FROM fragmentos WHERE documento_id = %s;", (doc_id,))
                    self._insert_chunks_in_batches(conn, doc_id, chunks, embedding_matrix)
                    return "actualizado"

                inserted = conn.execute(
                    """
                    INSERT INTO documentos (nombre_archivo, hash_sha256, total_paginas)
                    VALUES (%s, %s, %s)
                    RETURNING id;
                    """,
                    (nombre_archivo, hash_sha256, total_paginas),
                ).fetchone()
                doc_id = int(inserted["id"])
                self._insert_chunks_in_batches(conn, doc_id, chunks, embedding_matrix)
                return "insertado"

    def search_similar_chunks(
        self,
        query_embedding: Sequence[float] | np.ndarray,
        *,
        top_k: int = 5,
        documento_nombre: str | None = None,
    ) -> list[RetrievedChunk]:
        """Recupera los ``top_k`` fragmentos más similares por distancia coseno (`<=>`)."""
        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query_vec.shape[0] != self.embedding_dim:
            raise ValueError(
                f"El vector de consulta debe tener {self.embedding_dim} dimensiones; "
                f"recibidas {query_vec.shape[0]}."
            )

        safe_k = max(1, min(int(top_k), 50))
        filter_doc = (documento_nombre or "").strip()
        if filter_doc in {"", "Todos los documentos"}:
            filter_doc = ""

        with self._connect() as conn:
            with conn.transaction():
                # Mejora la recuperación en búsquedas filtradas sobre el índice HNSW.
                conn.execute("SET LOCAL hnsw.ef_search = 100;")
                if filter_doc:
                    rows = conn.execute(
                        """
                        SELECT
                            f.id AS fragmento_id,
                            d.nombre_archivo AS nombre_documento,
                            f.pagina_fisica,
                            f.indice_fragmento,
                            f.contenido,
                            (f.embedding <=> %s) AS distancia_coseno,
                            (1.0 - (f.embedding <=> %s)) AS similitud
                        FROM fragmentos f
                        JOIN documentos d ON d.id = f.documento_id
                        WHERE d.nombre_archivo = %s
                        ORDER BY f.embedding <=> %s ASC
                        LIMIT %s;
                        """,
                        (query_vec, query_vec, filter_doc, query_vec, safe_k),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT
                            f.id AS fragmento_id,
                            d.nombre_archivo AS nombre_documento,
                            f.pagina_fisica,
                            f.indice_fragmento,
                            f.contenido,
                            (f.embedding <=> %s) AS distancia_coseno,
                            (1.0 - (f.embedding <=> %s)) AS similitud
                        FROM fragmentos f
                        JOIN documentos d ON d.id = f.documento_id
                        ORDER BY f.embedding <=> %s ASC
                        LIMIT %s;
                        """,
                        (query_vec, query_vec, query_vec, safe_k),
                    ).fetchall()

        return [
            RetrievedChunk(
                fragmento_id=int(row["fragmento_id"]),
                nombre_documento=str(row["nombre_documento"]),
                pagina_fisica=int(row["pagina_fisica"]),
                indice_fragmento=int(row["indice_fragmento"]),
                contenido=str(row["contenido"]),
                similitud=float(row["similitud"]),
                distancia_coseno=float(row["distancia_coseno"]),
            )
            for row in rows
        ]

    def get_inventory_stats(self) -> InventoryStats:
        """Consulta en tiempo real los totales almacenados en PostgreSQL."""
        with self._connect(register_pgvector=False) as conn:
            doc_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_documentos,
                    COALESCE(SUM(total_paginas), 0) AS total_paginas
                FROM documentos;
                """
            ).fetchone()
            frag_row = conn.execute(
                "SELECT COUNT(*) AS total_fragmentos FROM fragmentos;"
            ).fetchone()

        return InventoryStats(
            total_documentos=int(doc_row["total_documentos"]) if doc_row else 0,
            total_fragmentos=int(frag_row["total_fragmentos"]) if frag_row else 0,
            total_paginas=int(doc_row["total_paginas"]) if doc_row else 0,
        )

    def list_documents(self) -> list[DocumentRecord]:
        """Devuelve el detalle de todos los documentos persistidos ordenados por fecha."""
        with self._connect(register_pgvector=False) as conn:
            rows = conn.execute(
                """
                SELECT
                    d.id,
                    d.nombre_archivo,
                    d.hash_sha256,
                    d.total_paginas,
                    d.fecha_ingesta,
                    COUNT(f.id) AS total_fragmentos
                FROM documentos d
                LEFT JOIN fragmentos f ON f.documento_id = d.id
                GROUP BY d.id
                ORDER BY d.fecha_ingesta DESC, d.nombre_archivo ASC;
                """
            ).fetchall()

        return [
            DocumentRecord(
                id=int(row["id"]),
                nombre_archivo=str(row["nombre_archivo"]),
                hash_sha256=str(row["hash_sha256"]),
                total_paginas=int(row["total_paginas"]),
                total_fragmentos=int(row["total_fragmentos"]),
                fecha_ingesta=row["fecha_ingesta"],
            )
            for row in rows
        ]

    def get_sample_chunks(self, nombre_archivo: str, limit: int = 5) -> list[dict]:
        """Obtiene una muestra ordenada de fragmentos de un documento para inspección."""
        with self._connect(register_pgvector=False) as conn:
            rows = conn.execute(
                """
                SELECT f.pagina_fisica, f.indice_fragmento, f.contenido
                FROM fragmentos f
                JOIN documentos d ON d.id = f.documento_id
                WHERE d.nombre_archivo = %s
                ORDER BY f.pagina_fisica ASC, f.indice_fragmento ASC
                LIMIT %s;
                """,
                (nombre_archivo, max(1, limit)),
            ).fetchall()
        return list(rows)

    def delete_document(self, nombre_archivo: str) -> bool:
        """Elimina un documento y sus fragmentos en cascada (`ON DELETE CASCADE`)."""
        with self._connect(register_pgvector=False) as conn:
            with conn.transaction():
                result = conn.execute(
                    "DELETE FROM documentos WHERE nombre_archivo = %s;",
                    (nombre_archivo,),
                )
        return (result.rowcount or 0) > 0

    def clear_all_documents(self) -> int:
        """Elimina todos los documentos y fragmentos de la base vectorial."""
        with self._connect(register_pgvector=False) as conn:
            with conn.transaction():
                result = conn.execute("DELETE FROM documentos;")
        return result.rowcount or 0
