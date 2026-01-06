import time
import streamlit as st

import sys
from pathlib import Path

# Asegura que el proyecto raíz está en el PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Backend real (cuando exista)
rag_import_error = None
try:
    from src.rag import rag_query  # recomendado
except Exception as e:
    rag_query = None
    rag_import_error = str(e)


# Demo stub (siempre disponible)
def rag_query_stub(question: str, k: int = 5) -> dict:
    if len(question.strip()) == 0:
        return {"answer": "Escribe una pregunta.", "rejected": True, "sources": []}

    # Ejemplo: simular rechazo si el usuario pregunta algo "fuera"
    if "asdasd" in question.lower():
        return {"answer": "No puedo responder con la base documental disponible.", "rejected": True, "sources": []}

    return {
        "answer": "Respuesta de ejemplo (conectar aquí al RAG real).",
        "rejected": False,
        "sources": [
            {
                "doc_id": "documento_ejemplo.pdf",
                "page": 1,
                "chunk_id": 12,
                "score": 0.82,
                "snippet": "Fragmento de ejemplo que justificaría la respuesta."
            }
        ]
    }

st.set_page_config(page_title="Chatbot RAG", layout="wide")
st.title("Chatbot documental (RAG)")
st.markdown("**Funcionalidad Extra:** Soporte Multilingüe (Español / English)")

with st.sidebar:
    st.header("Configuración")
    
    # --- NUEVO: Selector de Idioma (Funcionalidad Extra) ---
    lang_choice = st.radio("Idioma de Respuesta / Language", ["Español", "English"])
    target_lang = "es" if lang_choice == "Español" else "en"
    
    st.divider()
    
    st.header("Parámetros")
    k = st.slider("Top-K", 1, 10, 5)
    show_snippets = st.toggle("Mostrar fragmentos", value=True)

    mode = st.selectbox(
        "Modo",
        ["Demo (stub)", "RAG real"],
        index=1,
        help="Usa Demo mientras el backend se integra. Cambia a RAG real cuando esté listo."
    )
    st.divider()

    # --- NUEVO: Visualización de Métricas (Requisito del compañero) ---
    st.header("📊 Métricas del Sistema")
    st.info("Resultados sobre Ground Truth (15 preguntas):")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Recall@8", "1.00")
        st.metric("BERTScore", "0.81")
    with col2:
        st.metric("MRR", "0.856")
        st.metric("FactScore", "0.76")
    
    st.caption("Tiempo medio: 21.78s")
    st.divider()

    if rag_query is None:
        st.error(f"RAG real no conectado: {rag_import_error}")
    else:
        st.success("RAG real conectado ✅")


    if st.button("Limpiar chat"):
        st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []

def render_sources(sources):
    st.subheader("Fuentes")
    if not sources:
        st.info("No hay fuentes para mostrar.")
        return

    for i, s in enumerate(sources, start=1):
        doc_id = s.get("doc_id", "Documento")
        page = s.get("page", None)
        chunk_id = s.get("chunk_id", None)
        score = s.get("score", None)

        title = f"{i}) {doc_id}"
        if page is not None:
            title += f" — pág. {page}"
        if chunk_id is not None:
            title += f" — chunk {chunk_id}"
        if score is not None:
            try:
                title += f" — score: {float(score):.3f}"
            except Exception:
                title += f" — score: {score}"

        with st.expander(title):
            if show_snippets:
                st.write(s.get("snippet", ""))

# Placeholder dinámico según idioma
placeholder = "Escribe tu pregunta..." if target_lang == "es" else "Type your question here..."
question = st.chat_input(placeholder)

if question:
    st.session_state.history.append({"role": "user", "content": question})

    t0 = time.time()

    if mode == "RAG real":
        if rag_query is None:
            result = {
                "answer": "El backend RAG real aún no está conectado. Cambia el modo a Demo (stub).",
                "rejected": True,
                "sources": []
            }
        else:
            # --- NUEVO: Pasamos el target_lang al backend ---
            result = rag_query(question, k=k, target_lang=target_lang)
    else:
        result = rag_query_stub(question, k=k)

    dt = time.time() - t0

    st.session_state.history.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "meta": {
            "rejected": result.get("rejected", False),
            "sources": result.get("sources", []),
            "latency": dt,
            "k": k
        }
    })

for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])

            meta = msg.get("meta", {})
            st.caption(
                f"Top-K: {meta.get('k', '?')} · "
                f"Fuentes: {len(meta.get('sources', []))} · "
                f"Tiempo: {meta.get('latency', 0):.2f}s"
            )

            if meta.get("rejected", False):
                warn_msg = "No se encontró evidencia suficiente." if target_lang == "es" else "Not enough evidence found."
                st.warning(warn_msg)

            render_sources(meta.get("sources", []))