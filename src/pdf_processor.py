"""Extracción eficiente de texto página por página y fragmentación con PyMuPDF."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import pymupdf as fitz  # PyMuPDF

logger = logging.getLogger(__name__)

_HORIZONTAL_WS_RE = re.compile(r"[ \t\f\v]+")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")
_SPLIT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass(frozen=True)
class PDFInput:
    """Entrada de documento PDF en memoria lista para procesar."""

    nombre_archivo: str
    contenido_bytes: bytes


@dataclass(frozen=True)
class TextChunk:
    """Fragmento de texto con trazabilidad exacta de documento y página física."""

    nombre_documento: str
    pagina_fisica: int
    indice_fragmento: int
    contenido: str


@dataclass(frozen=True)
class PageReadError:
    """Error capturado durante la lectura de un archivo o página individual."""

    nombre_documento: str
    pagina_fisica: int | None
    mensaje: str


@dataclass
class DocumentDiagnostic:
    """Reporte diagnóstico completo de la ingesta de un PDF."""

    nombre_archivo: str
    hash_sha256: str
    total_paginas: int = 0
    paginas_procesadas: int = 0
    paginas_con_texto: int = 0
    paginas_sin_texto: list[int] = field(default_factory=list)
    errores: list[PageReadError] = field(default_factory=list)
    total_fragmentos: int = 0
    estado_persistencia: str = "pendiente"
    tiempo_segundos: float = 0.0

    @property
    def tiene_errores_archivo(self) -> bool:
        """Indica si ocurrió un error que impidió abrir el archivo completo."""
        return any(err.pagina_fisica is None for err in self.errores)


@dataclass
class ProcessedDocument:
    """Resultado de procesar un documento PDF: diagnóstico y fragmentos."""

    diagnostic: DocumentDiagnostic
    chunks: list[TextChunk]


ProgressCallback = Callable[[int, int, str], None]


def compute_sha256_bytes(data: bytes) -> str:
    """Calcula el hash SHA-256 hexadecimal para contenido binario en memoria."""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo en bloques de 1 MiB."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_page_text(raw_text: str) -> str:
    """Limpia caracteres nulos y espacios superfluos conservando saltos útiles.

    PostgreSQL rechaza bytes NUL (`\\x00`) en columnas ``TEXT``, por lo que se
    eliminan explícitamente antes de fragmentar.
    """
    if not raw_text:
        return ""
    sanitized = raw_text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = [_HORIZONTAL_WS_RE.sub(" ", line).strip() for line in sanitized.split("\n")]
    joined = "\n".join(cleaned_lines)
    joined = _EXCESS_NEWLINES_RE.sub("\n\n", joined)
    return joined.strip()


def _find_split_end(text: str, start: int, max_end: int) -> int:
    """Busca un límite natural de corte hacia atrás dentro de la segunda mitad."""
    min_split = start + (max_end - start) // 2
    window = text[min_split:max_end]
    for sep in _SPLIT_SEPARATORS:
        idx = window.rfind(sep)
        if idx != -1:
            candidate = min_split + idx + len(sep)
            if candidate > start:
                return candidate
    return max_end


def _align_next_start(text: str, candidate_start: int, prev_start: int, prev_end: int) -> int:
    """Alinea el inicio del siguiente fragmento a un límite de palabra sin retroceder."""
    next_start = max(candidate_start, prev_start + 1)
    if next_start <= 0 or next_start >= prev_end:
        return min(next_start, prev_end)
    if text[next_start - 1].isspace():
        return next_start
    # Avanza unos caracteres hasta el siguiente espacio para no partir una palabra.
    lookahead_limit = min(prev_end, next_start + 40)
    space_idx = text.find(" ", next_start, lookahead_limit)
    if space_idx != -1 and space_idx + 1 < prev_end:
        return space_idx + 1
    return next_start


def split_text_into_chunks(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[str]:
    """Divide texto de una página física en fragmentos con solapamiento controlado.

    Args:
        text: Texto limpio extraído de una sola página física.
        chunk_size: Tamaño máximo objetivo de cada fragmento en caracteres.
        chunk_overlap: Solapamiento aproximado entre fragmentos consecutivos.

    Returns:
        Lista ordenada de fragmentos no vacíos pertenecientes a la página.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe estar entre 0 y chunk_size - 1.")

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    text_len = len(cleaned)
    start = 0

    while start < text_len:
        raw_end = min(start + chunk_size, text_len)
        end = _find_split_end(cleaned, start, raw_end) if raw_end < text_len else text_len

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_len:
            break

        raw_next_start = end - chunk_overlap
        start = _align_next_start(cleaned, raw_next_start, start, end)

    return chunks


def validate_pdf_batch_size(count: int, max_files: int = 3) -> None:
    """Valida que el lote de PDFs no esté vacío ni supere el máximo permitido."""
    if count <= 0:
        raise ValueError("Debe proporcionar al menos un archivo PDF para procesar.")
    if count > max_files:
        raise ValueError(
            f"Se recibieron {count} archivos PDF, pero el máximo permitido por lote es {max_files}."
        )


def discover_pdfs_in_directory(data_dir: Path | str, max_files: int = 3) -> list[Path]:
    """Lista los archivos PDF disponibles en una carpeta ordenados alfabéticamente.

    Args:
        data_dir: Ruta de la carpeta a inspeccionar (ej. ``data/``).
        max_files: Límite máximo de archivos permitido si se valida posteriormente.

    Returns:
        Lista ordenada de rutas a archivos ``.pdf``.
    """
    directory = Path(data_dir)
    if not directory.exists() or not directory.is_dir():
        return []
    pdf_paths = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )
    if max_files > 0 and len(pdf_paths) > max_files:
        logger.warning(
            "La carpeta %s contiene %d PDFs; supera el límite recomendado de %d.",
            directory,
            len(pdf_paths),
            max_files,
        )
    return pdf_paths


def _extract_from_fitz_document(
    doc: fitz.Document,
    nombre_archivo: str,
    hash_sha256: str,
    chunk_size: int,
    chunk_overlap: int,
    progress_callback: ProgressCallback | None = None,
) -> ProcessedDocument:
    """Itera página por página sobre un documento PyMuPDF abierto."""
    started = time.perf_counter()
    diagnostic = DocumentDiagnostic(nombre_archivo=nombre_archivo, hash_sha256=hash_sha256)
    chunks: list[TextChunk] = []

    if doc.is_encrypted and not doc.authenticate(""):
        diagnostic.errores.append(
            PageReadError(
                nombre_documento=nombre_archivo,
                pagina_fisica=None,
                mensaje="El archivo PDF está protegido con contraseña y no puede leerse.",
            )
        )
        diagnostic.tiempo_segundos = time.perf_counter() - started
        return ProcessedDocument(diagnostic=diagnostic, chunks=chunks)

    total_pages = doc.page_count
    diagnostic.total_paginas = total_pages
    if total_pages == 0:
        diagnostic.errores.append(
            PageReadError(
                nombre_documento=nombre_archivo,
                pagina_fisica=None,
                mensaje="El archivo PDF no contiene páginas.",
            )
        )
        diagnostic.tiempo_segundos = time.perf_counter() - started
        return ProcessedDocument(diagnostic=diagnostic, chunks=chunks)

    next_chunk_index = 0
    progress_step = max(1, total_pages // 25)

    for zero_idx in range(total_pages):
        pagina_fisica = zero_idx + 1
        try:
            page = doc.load_page(zero_idx)
            raw_text = page.get_text("text")
            cleaned_text = normalize_page_text(raw_text)
        except Exception as exc:
            logger.exception(
                "Error leyendo página %d de %s: %s", pagina_fisica, nombre_archivo, exc
            )
            diagnostic.errores.append(
                PageReadError(
                    nombre_documento=nombre_archivo,
                    pagina_fisica=pagina_fisica,
                    mensaje=str(exc),
                )
            )
            continue

        diagnostic.paginas_procesadas += 1
        if not cleaned_text:
            diagnostic.paginas_sin_texto.append(pagina_fisica)
        else:
            diagnostic.paginas_con_texto += 1
            page_chunks = split_text_into_chunks(
                cleaned_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for piece in page_chunks:
                chunks.append(
                    TextChunk(
                        nombre_documento=nombre_archivo,
                        pagina_fisica=pagina_fisica,
                        indice_fragmento=next_chunk_index,
                        contenido=piece,
                    )
                )
                next_chunk_index += 1

        if progress_callback and (
            pagina_fisica == 1
            or pagina_fisica % progress_step == 0
            or pagina_fisica == total_pages
        ):
            progress_callback(pagina_fisica, total_pages, nombre_archivo)

    diagnostic.total_fragmentos = len(chunks)
    diagnostic.tiempo_segundos = time.perf_counter() - started
    return ProcessedDocument(diagnostic=diagnostic, chunks=chunks)


def process_pdf_bytes(
    nombre_archivo: str,
    contenido_bytes: bytes,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    progress_callback: ProgressCallback | None = None,
) -> ProcessedDocument:
    """Procesa un PDF desde bytes en memoria capturando errores sin abortar el lote."""
    started = time.perf_counter()
    clean_name = Path(nombre_archivo).name
    sha256 = compute_sha256_bytes(contenido_bytes)

    if not contenido_bytes:
        diag = DocumentDiagnostic(
            nombre_archivo=clean_name,
            hash_sha256=sha256,
            errores=[
                PageReadError(
                    nombre_documento=clean_name,
                    pagina_fisica=None,
                    mensaje="El archivo PDF está vacío (0 bytes).",
                )
            ],
            tiempo_segundos=time.perf_counter() - started,
        )
        return ProcessedDocument(diagnostic=diag, chunks=[])

    doc: fitz.Document | None = None
    try:
        doc = fitz.open(stream=contenido_bytes, filetype="pdf")
        return _extract_from_fitz_document(
            doc=doc,
            nombre_archivo=clean_name,
            hash_sha256=sha256,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        logger.exception("Error abriendo PDF en memoria '%s': %s", clean_name, exc)
        diag = DocumentDiagnostic(
            nombre_archivo=clean_name,
            hash_sha256=sha256,
            errores=[
                PageReadError(
                    nombre_documento=clean_name,
                    pagina_fisica=None,
                    mensaje=f"No se pudo abrir el archivo PDF: {exc}",
                )
            ],
            tiempo_segundos=time.perf_counter() - started,
        )
        return ProcessedDocument(diagnostic=diag, chunks=[])
    finally:
        if doc is not None:
            doc.close()


def process_pdf_file(
    file_path: Path | str,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    progress_callback: ProgressCallback | None = None,
) -> ProcessedDocument:
    """Procesa un PDF directamente desde disco minimizando uso de RAM en 300+ páginas."""
    started = time.perf_counter()
    path = Path(file_path)
    clean_name = path.name

    if not path.exists() or not path.is_file():
        diag = DocumentDiagnostic(
            nombre_archivo=clean_name,
            hash_sha256="",
            errores=[
                PageReadError(
                    nombre_documento=clean_name,
                    pagina_fisica=None,
                    mensaje=f"Archivo no encontrado: {path}",
                )
            ],
            tiempo_segundos=time.perf_counter() - started,
        )
        return ProcessedDocument(diagnostic=diag, chunks=[])

    doc: fitz.Document | None = None
    try:
        sha256 = compute_sha256_file(path)
        doc = fitz.open(str(path))
        return _extract_from_fitz_document(
            doc=doc,
            nombre_archivo=clean_name,
            hash_sha256=sha256,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        logger.exception("Error abriendo archivo PDF '%s': %s", path, exc)
        diag = DocumentDiagnostic(
            nombre_archivo=clean_name,
            hash_sha256="",
            errores=[
                PageReadError(
                    nombre_documento=clean_name,
                    pagina_fisica=None,
                    mensaje=f"No se pudo abrir el archivo PDF: {exc}",
                )
            ],
            tiempo_segundos=time.perf_counter() - started,
        )
        return ProcessedDocument(diagnostic=diag, chunks=[])
    finally:
        if doc is not None:
            doc.close()


def process_pdf_batch(
    inputs: Sequence[PDFInput | Path | str],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    max_files: int = 3,
    progress_callback: ProgressCallback | None = None,
) -> list[ProcessedDocument]:
    """Procesa un lote de hasta ``max_files`` PDFs continuando ante errores individuales."""
    validate_pdf_batch_size(len(inputs), max_files=max_files)
    results: list[ProcessedDocument] = []
    for item in inputs:
        if isinstance(item, PDFInput):
            processed = process_pdf_bytes(
                item.nombre_archivo,
                item.contenido_bytes,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                progress_callback=progress_callback,
            )
        else:
            processed = process_pdf_file(
                item,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                progress_callback=progress_callback,
            )
        results.append(processed)
    return results
