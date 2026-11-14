"""Pruebas unitarias para persistencia idempotente y búsqueda vectorial en VectorDatabase."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from src.database import VectorDatabase
from src.pdf_processor import TextChunk


class _FakeCursor:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn
        self._rows: list[dict[str, Any]] = []
        self.rowcount = 0

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def executemany(self, sql: str, seq_of_params: list[tuple[Any, ...]]) -> None:
        for params in seq_of_params:
            self._conn.execute(sql, params)

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _FakeConnection:
    """Simula tablas PostgreSQL (`documentos` y `fragmentos`) y operador `<=>`."""

    def __init__(self, store: _FakeStore) -> None:
        self.store = store

    @contextmanager
    def transaction(self):  # type: ignore[no-untyped-def]
        snapshot_docs = [dict(d) for d in self.store.documentos]
        snapshot_frags = [dict(f) for f in self.store.fragmentos]
        snapshot_doc_seq = self.store.doc_seq
        snapshot_frag_seq = self.store.frag_seq
        try:
            yield
        except Exception:
            self.store.documentos = snapshot_docs
            self.store.fragmentos = snapshot_frags
            self.store.doc_seq = snapshot_doc_seq
            self.store.frag_seq = snapshot_frag_seq
            raise

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _FakeCursor:
        cur = _FakeCursor(self)
        normalized = " ".join(sql.split())
        self.store.executed_sql.append(normalized)

        if normalized.startswith("CREATE EXTENSION") or normalized.startswith("CREATE TABLE") or normalized.startswith("CREATE INDEX") or normalized.startswith("SET LOCAL"):
            return cur

        if "FROM pg_extension" in normalized:
            cur._rows = [{"extname": "vector", "extversion": "0.8.0"}]
            return cur

        if "SELECT id, hash_sha256, total_paginas FROM documentos WHERE nombre_archivo = %s" in normalized:
            nombre = params[0]
            cur._rows = [
                {
                    "id": d["id"],
                    "hash_sha256": d["hash_sha256"],
                    "total_paginas": d["total_paginas"],
                }
                for d in self.store.documentos
                if d["nombre_archivo"] == nombre
            ]
            return cur

        if "SELECT COUNT(*) AS cnt FROM fragmentos WHERE documento_id = %s" in normalized:
            doc_id = params[0]
            cnt = sum(1 for f in self.store.fragmentos if f["documento_id"] == doc_id)
            cur._rows = [{"cnt": cnt}]
            return cur

        if "SELECT d.hash_sha256, d.total_paginas, COUNT(f.id) AS fragment_count" in normalized:
            nombre = params[0]
            doc = next((d for d in self.store.documentos if d["nombre_archivo"] == nombre), None)
            if doc:
                cnt = sum(1 for f in self.store.fragmentos if f["documento_id"] == doc["id"])
                cur._rows = [
                    {
                        "hash_sha256": doc["hash_sha256"],
                        "total_paginas": doc["total_paginas"],
                        "fragment_count": cnt,
                    }
                ]
            return cur

        if normalized.startswith("INSERT INTO documentos"):
            nombre, sha, total_paginas = params
            self.store.doc_seq += 1
            new_doc = {
                "id": self.store.doc_seq,
                "nombre_archivo": nombre,
                "hash_sha256": sha,
                "total_paginas": total_paginas,
                "fecha_ingesta": datetime.now(timezone.utc),
            }
            self.store.documentos.append(new_doc)
            cur._rows = [{"id": new_doc["id"]}]
            return cur

        if normalized.startswith("UPDATE documentos SET hash_sha256 = %s"):
            sha, total_paginas, doc_id = params
            for d in self.store.documentos:
                if d["id"] == doc_id:
                    d["hash_sha256"] = sha
                    d["total_paginas"] = total_paginas
                    d["fecha_ingesta"] = datetime.now(timezone.utc)
            return cur

        if normalized.startswith("UPDATE documentos SET fecha_ingesta = NOW()"):
            target = params[0]
            for d in self.store.documentos:
                if d["id"] == target or d["nombre_archivo"] == target:
                    d["fecha_ingesta"] = datetime.now(timezone.utc)
            return cur

        if normalized.startswith("DELETE FROM fragmentos WHERE documento_id = %s"):
            doc_id = params[0]
            self.store.fragmentos = [
                f for f in self.store.fragmentos if f["documento_id"] != doc_id
            ]
            return cur

        if normalized.startswith("INSERT INTO fragmentos"):
            doc_id, pagina_fisica, indice_fragmento, contenido, embedding = params
            self.store.frag_seq += 1
            self.store.fragmentos.append(
                {
                    "id": self.store.frag_seq,
                    "documento_id": doc_id,
                    "pagina_fisica": pagina_fisica,
                    "indice_fragmento": indice_fragmento,
                    "contenido": contenido,
                    "embedding": np.asarray(embedding, dtype=np.float32),
                }
            )
            return cur

        if "ORDER BY f.embedding <=> %s ASC" in normalized:
            if "WHERE d.nombre_archivo = %s" in normalized:
                q_vec, _, filter_doc, _, limit = params
            else:
                q_vec, _, _, limit = params
                filter_doc = None

            q_vec = np.asarray(q_vec, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec) or 1.0
            docs_by_id = {d["id"]: d for d in self.store.documentos}

            scored = []
            for f in self.store.fragmentos:
                doc = docs_by_id[f["documento_id"]]
                if filter_doc and doc["nombre_archivo"] != filter_doc:
                    continue
                v = f["embedding"]
                v_norm = np.linalg.norm(v) or 1.0
                cos_sim = float(np.dot(q_vec, v) / (q_norm * v_norm))
                cos_dist = 1.0 - cos_sim
                scored.append(
                    {
                        "fragmento_id": f["id"],
                        "nombre_documento": doc["nombre_archivo"],
                        "pagina_fisica": f["pagina_fisica"],
                        "indice_fragmento": f["indice_fragmento"],
                        "contenido": f["contenido"],
                        "distancia_coseno": cos_dist,
                        "similitud": cos_sim,
                    }
                )
            scored.sort(key=lambda r: r["distancia_coseno"])
            cur._rows = scored[: int(limit)]
            return cur

        if "COUNT(*) AS total_documentos" in normalized:
            cur._rows = [
                {
                    "total_documentos": len(self.store.documentos),
                    "total_paginas": sum(d["total_paginas"] for d in self.store.documentos),
                }
            ]
            return cur

        if "COUNT(*) AS total_fragmentos" in normalized:
            cur._rows = [{"total_fragmentos": len(self.store.fragmentos)}]
            return cur

        if "FROM documentos d LEFT JOIN fragmentos f" in normalized:
            rows = []
            for d in self.store.documentos:
                cnt = sum(1 for f in self.store.fragmentos if f["documento_id"] == d["id"])
                rows.append({**d, "total_fragmentos": cnt})
            cur._rows = rows
            return cur

        if normalized.startswith("DELETE FROM documentos WHERE nombre_archivo = %s"):
            nombre = params[0]
            target_ids = {d["id"] for d in self.store.documentos if d["nombre_archivo"] == nombre}
            before = len(self.store.documentos)
            self.store.documentos = [d for d in self.store.documentos if d["id"] not in target_ids]
            self.store.fragmentos = [
                f for f in self.store.fragmentos if f["documento_id"] not in target_ids
            ]
            cur.rowcount = before - len(self.store.documentos)
            return cur

        raise AssertionError(f"SQL no simulado en test: {normalized}")


class _FakeStore:
    def __init__(self) -> None:
        self.documentos: list[dict[str, Any]] = []
        self.fragmentos: list[dict[str, Any]] = []
        self.doc_seq = 0
        self.frag_seq = 0
        self.executed_sql: list[str] = []


class _TestableVectorDatabase(VectorDatabase):
    def __init__(self) -> None:
        super().__init__("postgresql://fake:fake@localhost:5432/fake")
        self.store = _FakeStore()

    @contextmanager
    def _connect(self, *, register_pgvector: bool = True):  # type: ignore[override]
        yield _FakeConnection(self.store)


def _unit_vec(index: int, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[index] = 1.0
    return vec


def test_initialize_schema_creates_vector_tables_and_hnsw() -> None:
    db = _TestableVectorDatabase()
    db.initialize_schema()

    joined = "\n".join(db.store.executed_sql)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in joined
    assert "CREATE TABLE IF NOT EXISTS documentos" in joined
    assert "CREATE TABLE IF NOT EXISTS fragmentos" in joined
    assert "embedding vector(384) NOT NULL" in joined
    assert "USING hnsw (embedding vector_cosine_ops)" in joined


def test_strict_idempotency_no_duplicates_on_reingestion() -> None:
    db = _TestableVectorDatabase()
    db.initialize_schema()

    chunks = [
        TextChunk("Manual_Operaciones.pdf", 1, 0, "Regla de freno neumático."),
        TextChunk("Manual_Operaciones.pdf", 2, 1, "Control de velocidad en terminal."),
    ]
    embeddings = np.vstack([_unit_vec(0), _unit_vec(1)])

    action_1 = db.upsert_document_idempotent(
        nombre_archivo="Manual_Operaciones.pdf",
        hash_sha256="a" * 64,
        total_paginas=2,
        chunks=chunks,
        embeddings=embeddings,
    )
    assert action_1 == "insertado"
    assert db.get_inventory_stats().total_documentos == 1
    assert db.get_inventory_stats().total_fragmentos == 2

    # Reingesta idéntica -> sin cambios ni duplicación
    action_2 = db.upsert_document_idempotent(
        nombre_archivo="Manual_Operaciones.pdf",
        hash_sha256="a" * 64,
        total_paginas=2,
        chunks=chunks,
        embeddings=embeddings,
    )
    assert action_2 == "sin_cambios"
    assert db.get_inventory_stats().total_documentos == 1
    assert db.get_inventory_stats().total_fragmentos == 2

    # Reemplazo forzado o actualización de hash -> reemplaza atómicamente
    new_chunks = [
        TextChunk("Manual_Operaciones.pdf", 1, 0, "Nueva versión del manual."),
        TextChunk("Manual_Operaciones.pdf", 2, 1, "Sección actualizada."),
        TextChunk("Manual_Operaciones.pdf", 3, 2, "Anexo nuevo."),
    ]
    new_embeddings = np.vstack([_unit_vec(0), _unit_vec(1), _unit_vec(2)])

    action_3 = db.upsert_document_idempotent(
        nombre_archivo="Manual_Operaciones.pdf",
        hash_sha256="b" * 64,
        total_paginas=3,
        chunks=new_chunks,
        embeddings=new_embeddings,
    )
    assert action_3 == "actualizado"
    stats = db.get_inventory_stats()
    assert stats.total_documentos == 1
    assert stats.total_fragmentos == 3
    assert stats.total_paginas == 3


def test_search_all_and_filtered_by_document() -> None:
    db = _TestableVectorDatabase()
    db.upsert_document_idempotent(
        nombre_archivo="Manual_Operaciones.pdf",
        hash_sha256="1" * 64,
        total_paginas=1,
        chunks=[TextChunk("Manual_Operaciones.pdf", 10, 0, "Protocolo de salida")],
        embeddings=np.vstack([_unit_vec(0)]),
    )
    db.upsert_document_idempotent(
        nombre_archivo="Protocolo_Incidentes.pdf",
        hash_sha256="2" * 64,
        total_paginas=1,
        chunks=[TextChunk("Protocolo_Incidentes.pdf", 4, 0, "Atención de colisión")],
        embeddings=np.vstack([_unit_vec(1)]),
    )

    all_hits = db.search_similar_chunks(_unit_vec(1), top_k=5, documento_nombre=None)
    assert len(all_hits) == 2
    assert all_hits[0].nombre_documento == "Protocolo_Incidentes.pdf"
    assert all_hits[0].pagina_fisica == 4
    assert all_hits[0].similitud == pytest.approx(1.0)

    filtered_hits = db.search_similar_chunks(
        _unit_vec(1),
        top_k=5,
        documento_nombre="Manual_Operaciones.pdf",
    )
    assert len(filtered_hits) == 1
    assert filtered_hits[0].nombre_documento == "Manual_Operaciones.pdf"


def test_delete_document_cascades_and_dimension_validation() -> None:
    db = _TestableVectorDatabase()
    db.upsert_document_idempotent(
        nombre_archivo="Mantenimiento.pdf",
        hash_sha256="3" * 64,
        total_paginas=1,
        chunks=[TextChunk("Mantenimiento.pdf", 1, 0, "Cambio de aceite")],
        embeddings=np.vstack([_unit_vec(0)]),
    )
    assert db.delete_document("Mantenimiento.pdf") is True
    assert db.get_inventory_stats().total_documentos == 0
    assert db.get_inventory_stats().total_fragmentos == 0

    with pytest.raises(ValueError):
        db.search_similar_chunks(np.zeros(128, dtype=np.float32), top_k=5)
