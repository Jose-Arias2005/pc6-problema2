"""Pruebas unitarias para extracción de PDF con PyMuPDF, chunking y diagnóstico."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import pytest

from src.pdf_processor import (
    PDFInput,
    discover_pdfs_in_directory,
    normalize_page_text,
    process_pdf_batch,
    process_pdf_bytes,
    split_text_into_chunks,
    validate_pdf_batch_size,
)


def _create_sample_image_pixmap() -> fitz.Pixmap:
    """Genera un pixmap RGB mínimo para simular páginas escaneadas sin texto."""
    return fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)


def _build_mixed_pdf_bytes() -> bytes:
    """Construye un PDF de 5 páginas con texto, blanco y página escaneada."""
    doc = fitz.open()
    try:
        # Página 1: texto extenso (> 1300 chars) para forzar múltiples fragmentos
        p1 = doc.new_page()
        long_ops_text = (
            "Manual de Operaciones de Transporte Urbano.\n\n"
            + " ".join(
                f"Regla operativa {i}: verificar presión de neumáticos y sistema neumático de frenos antes de iniciar ruta."
                for i in range(1, 18)
            )
        )
        p1.insert_textbox(fitz.Rect(40, 40, 550, 780), long_ops_text, fontsize=10)

        # Página 2: completamente en blanco
        doc.new_page()

        # Página 3: imagen sin texto seleccionable (simula escaneo)
        p3 = doc.new_page()
        pix = _create_sample_image_pixmap()
        p3.insert_image(fitz.Rect(50, 50, 150, 150), pixmap=pix)

        # Página 4: texto corto de mantenimiento
        p4 = doc.new_page()
        p4.insert_textbox(
            fitz.Rect(40, 40, 550, 200),
            "Procedimiento de mantenimiento preventivo cada 10,000 kilómetros.",
            fontsize=11,
        )

        # Página 5: solo espacios en blanco
        p5 = doc.new_page()
        p5.insert_text((50, 50), "   ")

        return doc.tobytes()
    finally:
        doc.close()


def test_normalize_page_text_strips_null_bytes_and_excess_whitespace() -> None:
    raw = "Línea  uno\x00\r\n\r\n\r\n\tLínea\t\tdos   \n"
    cleaned = normalize_page_text(raw)
    assert "\x00" not in cleaned
    assert cleaned == "Línea uno\n\nLínea dos"


def test_split_text_into_chunks_overlap_and_bounds() -> None:
    text = " ".join(f"Oración número {i} sobre seguridad vial y frenado." for i in range(60))
    chunks = split_text_into_chunks(text, chunk_size=300, chunk_overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # Verifica solapamiento real entre fragmentos consecutivos
    assert chunks[0][-30:] in chunks[1] or any(
        word in chunks[1][:100] for word in chunks[0][-80:].split()
    )


def test_split_text_unbroken_token_always_terminates() -> None:
    unbroken = "X" * 2000
    chunks = split_text_into_chunks(unbroken, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 4
    assert all(len(c) <= 500 for c in chunks)


def test_split_text_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        split_text_into_chunks("hola", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        split_text_into_chunks("hola", chunk_size=200, chunk_overlap=200)


def test_process_pdf_bytes_physical_pages_and_empty_page_detection() -> None:
    pdf_bytes = _build_mixed_pdf_bytes()
    result = process_pdf_bytes(
        "Manual_Operaciones.pdf",
        pdf_bytes,
        chunk_size=500,
        chunk_overlap=100,
    )
    diag = result.diagnostic

    assert diag.nombre_archivo == "Manual_Operaciones.pdf"
    assert len(diag.hash_sha256) == 64
    assert diag.total_paginas == 5
    assert diag.paginas_procesadas == 5
    assert diag.paginas_con_texto == 2
    assert diag.paginas_sin_texto == [2, 3, 5]
    assert diag.errores == []
    assert diag.total_fragmentos == len(result.chunks)
    assert diag.total_fragmentos >= 3

    p1_chunks = [c for c in result.chunks if c.pagina_fisica == 1]
    p4_chunks = [c for c in result.chunks if c.pagina_fisica == 4]
    assert len(p1_chunks) >= 2
    assert len(p4_chunks) == 1
    assert all(c.nombre_documento == "Manual_Operaciones.pdf" for c in result.chunks)
    assert [c.indice_fragmento for c in result.chunks] == list(range(len(result.chunks)))


def test_process_pdf_batch_continues_after_corrupt_and_encrypted_files() -> None:
    valid_bytes = _build_mixed_pdf_bytes()

    # Crea un PDF cifrado con contraseña
    enc_doc = fitz.open()
    try:
        page = enc_doc.new_page()
        page.insert_text((50, 50), "Contenido protegido")
        encrypted_bytes = enc_doc.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            user_pw="secreto",
            owner_pw="dueno",
        )
    finally:
        enc_doc.close()

    batch = [
        PDFInput("corrupto.pdf", b"contenido-invalido-no-es-pdf"),
        PDFInput("cifrado.pdf", encrypted_bytes),
        PDFInput("valido.pdf", valid_bytes),
    ]
    results = process_pdf_batch(batch, max_files=3)

    assert len(results) == 3
    assert results[0].diagnostic.tiene_errores_archivo is True
    assert results[0].chunks == []

    assert results[1].diagnostic.tiene_errores_archivo is True
    assert results[1].chunks == []

    assert results[2].diagnostic.errores == []
    assert results[2].diagnostic.total_paginas == 5
    assert len(results[2].chunks) >= 3


def test_process_pdf_page_level_error_captured_without_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = fitz.open()
    try:
        for page_num in (1, 2, 3):
            p = doc.new_page()
            p.insert_text((50, 50), f"Contenido de la página {page_num}.")
        pdf_bytes = doc.tobytes()
    finally:
        doc.close()

    original_get_text = fitz.Page.get_text

    def faulty_get_text(self: fitz.Page, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.number == 1:  # página física 2
            raise RuntimeError("Fallo simulado al decodificar flujo de página")
        return original_get_text(self, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_text", faulty_get_text)

    result = process_pdf_bytes("Manual_Falla.pdf", pdf_bytes)
    diag = result.diagnostic

    assert diag.total_paginas == 3
    assert diag.paginas_procesadas == 2
    assert len(diag.errores) == 1
    assert diag.errores[0].pagina_fisica == 2
    assert [c.pagina_fisica for c in result.chunks] == [1, 3]


def test_validate_pdf_batch_size_and_discover_pdfs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_pdf_batch_size(0, max_files=3)
    with pytest.raises(ValueError):
        validate_pdf_batch_size(4, max_files=3)
    validate_pdf_batch_size(3, max_files=3)

    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "a.PDF").write_bytes(b"%PDF-1.4")
    (tmp_path / "notas.txt").write_text("ignorar", encoding="utf-8")

    found = discover_pdfs_in_directory(tmp_path, max_files=3)
    assert [p.name for p in found] == ["a.PDF", "b.pdf"]


def test_300_page_pdf_efficiency() -> None:
    doc = fitz.open()
    try:
        for i in range(1, 301):
            p = doc.new_page()
            p.insert_text((40, 40), f"Manual técnico página física {i}. Control de frenos y ruta.")
        pdf_bytes = doc.tobytes()
    finally:
        doc.close()

    result = process_pdf_bytes("Manual_300_Paginas.pdf", pdf_bytes)
    assert result.diagnostic.total_paginas == 300
    assert result.diagnostic.paginas_procesadas == 300
    assert result.diagnostic.total_fragmentos == 300
    assert result.diagnostic.tiempo_segundos < 10.0
