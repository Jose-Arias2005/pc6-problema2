"""Interfaz web Streamlit para el Sistema RAG de manuales de transporte."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.config import Settings
from src.database import DocumentRecord, InventoryStats, VectorDatabase
from src.pdf_processor import (
    DocumentDiagnostic,
    PDFInput,
    ProcessedDocument,
    discover_pdfs_in_directory,
    process_pdf_bytes,
    process_pdf_file,
    validate_pdf_batch_size,
)
from src.rag_engine import GeminiEmbedder, RAGResponse, RAGEngine

DATA_DIR = Path(__file__).resolve().parent / "data"

PERSISTENCE_LABELS: dict[str, tuple[str, str]] = {
    "insertado": ("Nuevo documento indexado", "success"),
    "actualizado": ("Actualizado atómicamente (sin duplicados)", "info"),
    "sin_cambios": ("Sin cambios (idempotente, omitida re-vectorización)", "info"),
    "error_archivo": ("Error al abrir archivo (no persistido)", "error"),
    "pendiente": ("Pendiente de persistencia", "warning"),
}

EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "¿Cuál es el procedimiento ante una falla de frenos o pérdida de presión neumática en ruta?",
    "¿Cada cuántos kilómetros o días se debe realizar el mantenimiento preventivo del motor?",
    "¿Qué pasos indica el protocolo de atención de incidentes en caso de accidente vial con pasajeros?",
)


st.set_page_config(
    page_title="RAG Manuales de Transporte",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_embedder(settings: Settings) -> GeminiEmbedder:
    """Construye el generador de embeddings con Gemini API."""
    return GeminiEmbedder(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_embedding_model,
        embedding_dim=settings.embedding_dim,
    )


def get_database(settings: Settings) -> VectorDatabase:
    """Crea la instancia de acceso a PostgreSQL + pgvector."""
    return VectorDatabase(dsn=settings.database_url, embedding_dim=settings.embedding_dim)


def ensure_database_ready(db: VectorDatabase) -> tuple[bool, str]:
    """Inicializa el esquema `pgvector` una vez conectado con éxito."""
    if not st.session_state.get("db_schema_initialized"):
        try:
            db.initialize_schema()
            st.session_state["db_schema_initialized"] = True
        except Exception as exc:
            st.session_state["db_schema_initialized"] = False
            return False, str(exc)

    ok, detail = db.health_check()
    if not ok:
        st.session_state["db_schema_initialized"] = False
    return ok, detail


def format_page_list(pages: list[int], max_preview: int = 40) -> str:
    """Formatea una lista de números de página física de forma legible."""
    if not pages:
        return "Ninguna"
    if len(pages) <= max_preview:
        return ", ".join(str(p) for p in pages)
    head = ", ".join(str(p) for p in pages[:max_preview])
    return f"{head} ... (+{len(pages) - max_preview} páginas más)"


def render_sidebar(
    settings: Settings,
    db: VectorDatabase,
    db_ok: bool,
    db_detail: str,
    documents: list[DocumentRecord],
    stats: InventoryStats,
) -> dict[str, Any]:
    """Renderiza contadores en tiempo real y controles de consulta en la barra lateral."""
    with st.sidebar:
        st.title("🚌 RAG Transporte")
        st.caption("Consulta inteligente sobre manuales operativos, mantenimiento e incidentes.")

        st.subheader("Estado de PostgreSQL + pgvector")
        if db_ok:
            st.success(db_detail)
        else:
            st.error(f"Error de conexión: {db_detail}")
            st.info("Ejecuta `docker compose up -d` en la raíz del proyecto y reintenta.")
            if st.button("🔄 Reintentar conexión", width="stretch"):
                st.session_state["db_schema_initialized"] = False
                st.rerun()

        st.divider()
        st.subheader("📊 Inventario en Tiempo Real")
        col1, col2 = st.columns(2)
        col1.metric("Documentos", stats.total_documentos)
        col2.metric("Fragmentos", stats.total_fragmentos)
        st.metric("Páginas registradas", stats.total_paginas)

        if documents:
            with st.expander(f"Documentos en BD ({len(documents)})", expanded=False):
                for doc in documents:
                    st.markdown(
                        f"- **{doc.nombre_archivo}**  \n"
                        f"  `{doc.total_paginas}` págs. · `{doc.total_fragmentos}` frags."
                    )

        st.divider()
        st.subheader("⚙️ Parámetros de Consulta")

        provider_options = ["gemini", "ollama"]
        default_provider_idx = provider_options.index(settings.llm_provider)
        selected_provider = st.selectbox(
            "Proveedor LLM",
            options=provider_options,
            index=default_provider_idx,
            format_func=lambda p: (
                f"Gemini ({settings.gemini_model})"
                if p == "gemini"
                else f"Ollama local ({settings.ollama_model})"
            ),
        )

        if selected_provider == "gemini" and settings.gemini_model_warning:
            st.warning(settings.gemini_model_warning)

        doc_names = [doc.nombre_archivo for doc in documents]
        scope_options = ["Todos los documentos"] + doc_names
        selected_scope = st.selectbox(
            "Alcance de búsqueda",
            options=scope_options,
            index=0,
            help="Consulta los manuales juntos o filtra por un PDF específico.",
        )

        top_k = st.slider(
            "Fragmentos a recuperar (top_k)",
            min_value=3,
            max_value=8,
            value=settings.default_top_k,
            step=1,
        )

        st.divider()
        st.caption(
            f"Embeddings: `{settings.gemini_embedding_model}` (384 dims · Gemini API)\n\n"
            f"Chunks: `{settings.chunk_size}` chars · Overlap: `{settings.chunk_overlap}`"
        )

    return {
        "provider": selected_provider,
        "scope": selected_scope,
        "top_k": top_k,
    }


def render_rag_response_card(resp: RAGResponse) -> None:
    """Muestra la respuesta generada, control de evidencia y pasajes exactos desplegables."""
    with st.chat_message("user"):
        st.markdown(resp.pregunta)

    with st.chat_message("assistant"):
        scope_label = resp.filtro_documento or "Todos los documentos"
        if resp.sin_evidencia:
            st.warning("Evidencia insuficiente en los documentos consultados")
        else:
            st.caption(
                f"✅ Respuesta basada en evidencia · Alcance: **{scope_label}** · "
                f"Modelo: `{resp.proveedor_usado}/{resp.modelo_usado}` · "
                f"Búsqueda: `{resp.tiempo_busqueda_ms:.0f} ms` · "
                f"Generación: `{resp.tiempo_generacion_ms:.0f} ms`"
            )

        if resp.advertencia_modelo:
            st.info(resp.advertencia_modelo)

        st.markdown(resp.respuesta)

        expander_label = (
            f"📄 Pasajes exactos utilizados ({len(resp.pasajes)} fragmentos recuperados)"
        )
        with st.expander(expander_label, expanded=False):
            if not resp.pasajes:
                st.info("No se recuperaron fragmentos desde PostgreSQL para esta consulta.")
            else:
                for idx, chunk in enumerate(resp.pasajes, start=1):
                    sim_pct = max(0.0, min(chunk.similitud * 100.0, 100.0))
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([3, 1.5, 1.5])
                        c1.markdown(f"**#{idx} · {chunk.nombre_documento}**")
                        c2.markdown(f"Página física: **{chunk.pagina_fisica}**")
                        c3.markdown(
                            f"Similitud: **{chunk.similitud:.4f}** ({sim_pct:.1f}%)"
                        )
                        st.caption(
                            f"Índice de fragmento: {chunk.indice_fragmento} · "
                            f"Distancia coseno (`<=>`): {chunk.distancia_coseno:.4f}"
                        )
                        st.text(chunk.contenido)


def render_query_tab(
    settings: Settings,
    db: VectorDatabase,
    db_ok: bool,
    stats: InventoryStats,
    query_cfg: dict[str, Any],
) -> None:
    """Pestaña principal de consultas semánticas y generación con citas."""
    st.subheader("💬 Consulta de Manuales con Evidencia Citada")

    scope_label = query_cfg["scope"]
    st.caption(
        f"Consultando: **{scope_label}** | `top_k = {query_cfg['top_k']}` | "
        f"Proveedor LLM: **{query_cfg['provider']}**"
    )

    if not db_ok:
        st.warning("Conecta PostgreSQL + pgvector antes de realizar consultas.")
        return

    if stats.total_fragmentos == 0:
        st.info(
            "Aún no hay fragmentos indexados en PostgreSQL. Dirígete a la pestaña "
            "**📥 Ingesta de PDFs y Diagnóstico** para cargar hasta 3 manuales PDF."
        )

    st.markdown("**Preguntas rápidas de prueba:**")
    q_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for idx, sample_q in enumerate(EXAMPLE_QUESTIONS):
        if q_cols[idx].button(
            f"Ejemplo {idx + 1}",
            help=sample_q,
            width="stretch",
            key=f"sample_q_{idx}",
        ):
            st.session_state["question_draft"] = sample_q

    with st.form("rag_query_form", clear_on_submit=False):
        question_input = st.text_area(
            "Escribe tu pregunta en español sobre operación, mantenimiento o incidentes:",
            key="question_draft",
            height=90,
            placeholder="Ej.: ¿Cuál es el protocolo de aislamiento y evacuación ante derrame de combustible?",
        )
        col_submit, col_clear = st.columns([3, 1])
        submitted = col_submit.form_submit_button(
            "🔎 Consultar manuales",
            type="primary",
            width="stretch",
        )
        clear_history = col_clear.form_submit_button(
            "🧹 Limpiar historial",
            width="stretch",
        )

    if clear_history:
        st.session_state["rag_history"] = []
        st.rerun()

    if submitted:
        if not question_input or not question_input.strip():
            st.warning("Por favor escribe una pregunta válida.")
        else:
            try:
                embedder = get_embedder(settings)
                engine = RAGEngine(db=db, embedder=embedder, settings=settings)
                with st.spinner("Generando embeddings con Gemini API, buscando en pgvector y redactando respuesta..."):
                    rag_response = engine.answer_question(
                        question_input,
                        top_k=query_cfg["top_k"],
                        documento_nombre=query_cfg["scope"],
                        provider_override=query_cfg["provider"],
                    )
                history: list[RAGResponse] = st.session_state.setdefault("rag_history", [])
                history.insert(0, rag_response)
            except Exception as exc:
                st.error(f"No se pudo completar la consulta: {exc}")

    history = st.session_state.get("rag_history", [])
    if history:
        st.divider()
        for item in history:
            render_rag_response_card(item)


def render_diagnostic_summary(diagnostics: list[DocumentDiagnostic]) -> None:
    """Despliega el reporte visual obligatorio posterior a la ingesta."""
    if not diagnostics:
        return

    st.divider()
    st.subheader("📋 Reporte de Diagnóstico de Ingesta")

    total_docs = len(diagnostics)
    total_pages_processed = sum(d.paginas_procesadas for d in diagnostics)
    total_empty_pages = sum(len(d.paginas_sin_texto) for d in diagnostics)
    total_errors = sum(len(d.errores) for d in diagnostics)
    total_chunks = sum(d.total_fragmentos for d in diagnostics)
    total_time = sum(d.tiempo_segundos for d in diagnostics)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("PDFs procesados", total_docs)
    m2.metric("Páginas procesadas", total_pages_processed)
    m3.metric("Páginas sin texto", total_empty_pages)
    m4.metric("Errores capturados", total_errors)
    m5.metric("Fragmentos generados", total_chunks)
    m6.metric("Tiempo extracción", f"{total_time:.2f} s")

    for diag in diagnostics:
        status_text, status_kind = PERSISTENCE_LABELS.get(
            diag.estado_persistencia,
            (diag.estado_persistencia, "info"),
        )
        header = (
            f"📘 {diag.nombre_archivo} — {diag.paginas_procesadas}/{diag.total_paginas} págs. · "
            f"{diag.total_fragmentos} fragmentos · {status_text}"
        )
        with st.expander(header, expanded=True):
            if status_kind == "success":
                st.success(f"Estado en PostgreSQL: **{status_text}**")
            elif status_kind == "error":
                st.error(f"Estado en PostgreSQL: **{status_text}**")
            elif status_kind == "warning":
                st.warning(f"Estado en PostgreSQL: **{status_text}**")
            else:
                st.info(f"Estado en PostgreSQL: **{status_text}**")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total páginas PDF", diag.total_paginas)
            c2.metric("Páginas procesadas", diag.paginas_procesadas)
            c3.metric("Con texto", diag.paginas_con_texto)
            c4.metric("Sin texto", len(diag.paginas_sin_texto))
            c5.metric("Fragmentos", diag.total_fragmentos)

            st.caption(
                f"SHA-256: `{diag.hash_sha256 or 'N/A'}` · "
                f"Duración lectura PDF: `{diag.tiempo_segundos:.2f} s`"
            )

            st.markdown("**Páginas sin texto seleccionable (en blanco o escaneadas como imagen):**")
            if diag.paginas_sin_texto:
                st.warning(
                    f"Se detectaron **{len(diag.paginas_sin_texto)}** página(s) sin texto "
                    f"seleccionable: `[{format_page_list(diag.paginas_sin_texto, max_preview=100)}]`"
                )
            else:
                st.success("Ninguna. Todas las páginas procesadas contienen texto seleccionable.")

            st.markdown("**Errores de lectura por archivo o página:**")
            if diag.errores:
                for err in diag.errores:
                    ubicacion = (
                        f"Página física {err.pagina_fisica}"
                        if err.pagina_fisica is not None
                        else "Archivo completo"
                    )
                    st.error(f"**{ubicacion}:** {err.mensaje}")
            else:
                st.success("0 errores de lectura durante el procesamiento.")


def _run_ingestion_pipeline(
    *,
    settings: Settings,
    db: VectorDatabase,
    source_items: list[PDFInput | Path],
    chunk_size: int,
    chunk_overlap: int,
    force_replace: bool,
) -> list[DocumentDiagnostic]:
    """Ejecuta lectura PyMuPDF, embedding local por lotes y persistencia idempotente."""
    embedder = get_embedder(settings)
    engine = RAGEngine(db=db, embedder=embedder, settings=settings)

    progress_bar = st.progress(0, text="Iniciando procesamiento de documentos...")
    status_box = st.empty()
    diagnostics: list[DocumentDiagnostic] = []
    total_files = len(source_items)

    for idx, item in enumerate(source_items, start=1):
        file_label = item.nombre_archivo if isinstance(item, PDFInput) else item.name
        base_pct = int(((idx - 1) / total_files) * 100)
        file_span = 100 / total_files

        def on_pdf_progress(page_num: int, page_total: int, doc_name: str) -> None:
            pct = min(99, int(base_pct + (page_num / max(1, page_total)) * (file_span * 0.45)))
            progress_bar.progress(
                pct,
                text=f"[{idx}/{total_files}] Leyendo `{doc_name}` — pág. {page_num}/{page_total}",
            )

        status_box.info(f"Extrayendo texto por página física de **{file_label}**...")
        if isinstance(item, PDFInput):
            processed: ProcessedDocument = process_pdf_bytes(
                item.nombre_archivo,
                item.contenido_bytes,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                progress_callback=on_pdf_progress,
            )
        else:
            processed = process_pdf_file(
                item,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                progress_callback=on_pdf_progress,
            )

        def on_embed_progress(done_chunks: int, total_chunks: int) -> None:
            pct = min(
                99,
                int(base_pct + (file_span * 0.45) + (done_chunks / max(1, total_chunks)) * (file_span * 0.5)),
            )
            progress_bar.progress(
                pct,
                text=(
                    f"[{idx}/{total_files}] Generando embeddings con Gemini API "
                    f"(`{file_label}`) — {done_chunks}/{total_chunks} fragmentos"
                ),
            )

        status_box.info(
            f"Generando embeddings con Gemini API y persistiendo atómicamente "
            f"**{file_label}** en PostgreSQL (`pgvector`)..."
        )
        engine.persist_document(
            processed,
            force_replace=force_replace,
            progress_callback=on_embed_progress,
        )
        diagnostics.append(processed.diagnostic)
        progress_bar.progress(
            min(100, int((idx / total_files) * 100)),
            text=f"Completado [{idx}/{total_files}]: {file_label}",
        )

    status_box.empty()
    progress_bar.progress(100, text="Ingesta completada.")
    return diagnostics


def render_ingestion_tab(settings: Settings, db: VectorDatabase, db_ok: bool) -> None:
    """Pestaña de carga de hasta 3 PDFs (web o carpeta `data/`) y diagnóstico."""
    st.subheader("📥 Ingesta de Manuales PDF y Reporte de Diagnóstico")
    st.markdown(
        "Carga hasta **3 archivos PDF** (hasta 300 páginas por manual, 900 páginas en total) "
        "desde la interfaz web o directamente desde la carpeta local `data/`."
    )

    if not db_ok:
        st.warning("Inicia PostgreSQL con `docker compose up -d` antes de indexar documentos.")
        return

    with st.expander("⚙️ Opciones de fragmentación e idempotencia", expanded=False):
        c1, c2, c3 = st.columns(3)
        chunk_size = c1.number_input(
            "Tamaño de fragmento (caracteres)",
            min_value=400,
            max_value=2000,
            value=settings.chunk_size,
            step=50,
        )
        chunk_overlap = c2.number_input(
            "Solapamiento (overlap)",
            min_value=0,
            max_value=int(chunk_size) - 50,
            value=min(settings.chunk_overlap, int(chunk_size) - 50),
            step=25,
        )
        force_replace = c3.checkbox(
            "Forzar reemplazo atómico",
            value=False,
            help=(
                "Si el archivo tiene el mismo nombre y SHA-256 ya guardado, por defecto se omite "
                "recalcular embeddings. Marca esta opción para forzar el reemplazo transaccional completo."
            ),
        )

    source_mode = st.radio(
        "Origen de los archivos PDF",
        options=["Subir archivos desde el navegador (máx. 3)", "Procesar carpeta local data/"],
        horizontal=True,
    )

    if source_mode.startswith("Subir"):
        uploaded_files = st.file_uploader(
            f"Selecciona hasta {settings.max_pdf_files} manuales en formato PDF",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and len(uploaded_files) > settings.max_pdf_files:
            st.error(
                f"Seleccionaste {len(uploaded_files)} archivos, pero el límite máximo por carga "
                f"es de {settings.max_pdf_files} PDFs."
            )
        elif uploaded_files:
            st.caption(
                "Archivos listos: " + ", ".join(f"`{f.name}`" for f in uploaded_files)
            )

        if st.button(
            "🚀 Procesar e indexar archivos subidos",
            type="primary",
            disabled=not uploaded_files or len(uploaded_files) > settings.max_pdf_files,
        ):
            try:
                validate_pdf_batch_size(len(uploaded_files), max_files=settings.max_pdf_files)
                pdf_inputs: list[PDFInput | Path] = [
                    PDFInput(nombre_archivo=f.name, contenido_bytes=f.getvalue())
                    for f in uploaded_files
                ]
                diagnostics = _run_ingestion_pipeline(
                    settings=settings,
                    db=db,
                    source_items=pdf_inputs,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    force_replace=bool(force_replace),
                )
                st.session_state["last_diagnostics"] = diagnostics
                st.rerun()
            except Exception as exc:
                st.error(f"Error durante la ingesta: {exc}")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        local_pdfs = discover_pdfs_in_directory(DATA_DIR, max_files=settings.max_pdf_files)
        if not local_pdfs:
            st.info(
                f"No hay archivos `.pdf` en `{DATA_DIR}`. Copia hasta "
                f"{settings.max_pdf_files} manuales allí y recarga la vista."
            )
        else:
            st.markdown(f"PDFs detectados en `{DATA_DIR.name}/` ({len(local_pdfs)}):")
            selected_names = st.multiselect(
                f"Selecciona hasta {settings.max_pdf_files} archivos a procesar",
                options=[p.name for p in local_pdfs],
                default=[p.name for p in local_pdfs[: settings.max_pdf_files]],
            )
            if len(selected_names) > settings.max_pdf_files:
                st.error(
                    f"Selecciona como máximo {settings.max_pdf_files} PDFs por lote."
                )

            if st.button(
                "📂 Procesar e indexar desde data/",
                type="primary",
                disabled=not selected_names or len(selected_names) > settings.max_pdf_files,
            ):
                try:
                    validate_pdf_batch_size(len(selected_names), max_files=settings.max_pdf_files)
                    by_name = {p.name: p for p in local_pdfs}
                    selected_paths: list[PDFInput | Path] = [
                        by_name[name] for name in selected_names
                    ]
                    diagnostics = _run_ingestion_pipeline(
                        settings=settings,
                        db=db,
                        source_items=selected_paths,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        force_replace=bool(force_replace),
                    )
                    st.session_state["last_diagnostics"] = diagnostics
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error procesando carpeta data/: {exc}")

    last_diagnostics: list[DocumentDiagnostic] = st.session_state.get("last_diagnostics", [])
    if last_diagnostics:
        render_diagnostic_summary(last_diagnostics)


def render_inventory_tab(
    db: VectorDatabase,
    db_ok: bool,
    documents: list[DocumentRecord],
    stats: InventoryStats,
) -> None:
    """Pestaña de inspección directa de persistencia en PostgreSQL."""
    st.subheader("🗄️ Inventario Persistente en PostgreSQL (`pgvector`)")
    if not db_ok:
        st.warning("PostgreSQL no está disponible actualmente.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total documentos en tabla `documentos`", stats.total_documentos)
    c2.metric("Total fragmentos en tabla `fragmentos`", stats.total_fragmentos)
    c3.metric("Suma de páginas físicas indexadas", stats.total_paginas)

    if not documents:
        st.info("La base de datos vectorial está vacía. Indexa manuales desde la pestaña de Ingesta.")
        return

    table_rows = [
        {
            "ID": doc.id,
            "Nombre del archivo": doc.nombre_archivo,
            "Páginas totales": doc.total_paginas,
            "Fragmentos": doc.total_fragmentos,
            "Hash SHA-256": f"{doc.hash_sha256[:16]}...",
            "Fecha ingesta": doc.fecha_ingesta.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for doc in documents
    ]
    try:
        st.dataframe(table_rows, width="stretch", hide_index=True)
    except Exception:
        header = "| ID | Nombre del archivo | Páginas totales | Fragmentos | Hash SHA-256 | Fecha ingesta |"
        sep = "| --- | --- | ---: | ---: | --- | --- |"
        lines = [
            f"| {r['ID']} | {r['Nombre del archivo']} | {r['Páginas totales']} | {r['Fragmentos']} | `{r['Hash SHA-256']}` | {r['Fecha ingesta']} |"
            for r in table_rows
        ]
        st.markdown("\n".join([header, sep, *lines]))

    st.divider()
    col_inspect, col_delete = st.columns(2)

    with col_inspect:
        inspect_target = st.selectbox(
            "Inspeccionar fragmentos almacenados de:",
            options=[doc.nombre_archivo for doc in documents],
            key="inspect_doc_select",
        )
        if inspect_target:
            samples = db.get_sample_chunks(inspect_target, limit=5)
            for sample in samples:
                with st.container(border=True):
                    st.caption(
                        f"Página física: {sample['pagina_fisica']} · "
                        f"Índice fragmento: {sample['indice_fragmento']}"
                    )
                    st.text(sample["contenido"])

    with col_delete:
        st.markdown("**Administración de registros**")
        delete_target = st.selectbox(
            "Eliminar documento individual (cascada a fragmentos):",
            options=[doc.nombre_archivo for doc in documents],
            key="delete_doc_select",
        )
        if st.button("🗑️ Eliminar documento seleccionado", width="stretch"):
            if db.delete_document(delete_target):
                st.success(f"Documento '{delete_target}' eliminado.")
                st.rerun()

        st.markdown("---")
        confirm_clear = st.checkbox("Confirmar vaciado total de la base vectorial")
        if st.button(
            "⚠️ Vaciar todos los documentos",
            disabled=not confirm_clear,
            width="stretch",
        ):
            deleted_count = db.clear_all_documents()
            st.session_state["last_diagnostics"] = []
            st.success(f"Se eliminaron {deleted_count} documentos y sus fragmentos.")
            st.rerun()


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    settings = Settings.from_env()
    db = get_database(settings)
    db_ok, db_detail = ensure_database_ready(db)

    if db_ok:
        stats = db.get_inventory_stats()
        documents = db.list_documents()
    else:
        stats = InventoryStats(total_documentos=0, total_fragmentos=0, total_paginas=0)
        documents = []

    query_cfg = render_sidebar(settings, db, db_ok, db_detail, documents, stats)

    st.title("📚 Sistema RAG para Manuales de Empresa de Transporte")
    st.markdown(
        "Consulta técnica con trazabilidad exacta a **documento y página física** sobre "
        "Manual de Operaciones, Procedimiento de Mantenimiento y Protocolo de Atención de Incidentes."
    )

    tab_query, tab_ingest, tab_inventory = st.tabs(
        [
            "💬 Consultas a los Manuales",
            "📥 Ingesta de PDFs y Diagnóstico",
            "🗄️ Inventario PostgreSQL",
        ]
    )

    with tab_query:
        render_query_tab(settings, db, db_ok, stats, query_cfg)

    with tab_ingest:
        render_ingestion_tab(settings, db, db_ok)

    with tab_inventory:
        render_inventory_tab(db, db_ok, documents, stats)


if __name__ == "__main__":
    main()
