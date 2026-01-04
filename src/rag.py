from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from functools import lru_cache

import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer


# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / ".artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BART_MODEL_NAME = "facebook/bart-large-cnn"
ES_EN_MODEL_NAME = "Helsinki-NLP/opus-mt-es-en"
EN_ES_MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"

# Umbral simple para “no hay doc relevante”
# Como usamos cosine similarity (IndexFlatIP), scores altos = mejor.
MIN_SCORE_TO_ANSWER = 0.30


# -----------------------------
# Types
# -----------------------------
@dataclass
class Chunk:
    doc_id: str
    page: Optional[int]
    chunk_id: str
    text: str


# -----------------------------
# Text extraction
# -----------------------------
def _extract_pdf_text_per_page(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Devuelve lista de (page_number_1_based, text).
    Intentamos PyMuPDF (fitz) si está, si no, pdfplumber.
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        pages = []
        for i in range(len(doc)):
            txt = doc.load_page(i).get_text("text") or ""
            pages.append((i + 1, txt))
        doc.close()
        return pages
    except Exception:
        pass

    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                txt = page.extract_text() or ""
                pages.append((i + 1, txt))
        return pages
    except Exception as e:
        raise RuntimeError(f"No pude leer el PDF {pdf_path.name}: {e}") from e


def _clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """
    Chunking simple por caracteres, con solape (mejor recuperación).
    Basado en vuestro notebook, pero con overlap para no cortar ideas.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


# -----------------------------
# Build / Load index
# -----------------------------
def _build_chunks_from_data_dir(data_dir: Path) -> List[Chunk]:
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    # 1) Prioridad: PDFs en data/raw
    pdfs = sorted(list(raw_dir.glob("*.pdf"))) if raw_dir.exists() else []
    if pdfs:
        all_chunks: List[Chunk] = []
        for pdf in pdfs:
            doc_id = pdf.name
            pages = _extract_pdf_text_per_page(pdf)
            for page_num, page_text in pages:
                page_text = _clean_text(page_text)
                for j, ch in enumerate(_chunk_text(page_text, chunk_size=1000, overlap=150)):
                    all_chunks.append(
                        Chunk(
                            doc_id=doc_id,
                            page=page_num,
                            chunk_id=f"{doc_id}:p{page_num}:c{j}",
                            text=ch,
                        )
                    )
        return all_chunks

    # 2) Alternativa: TXT en data/processed (si ya tenéis texto extraído)
    txts = sorted(list(processed_dir.glob("*.txt"))) if processed_dir.exists() else []
    if txts:
        all_chunks: List[Chunk] = []
        for txt in txts:
            doc_id = txt.name
            text = _clean_text(txt.read_text(encoding="utf-8", errors="ignore"))
            for j, ch in enumerate(_chunk_text(text, chunk_size=1000, overlap=150)):
                all_chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        page=None,
                        chunk_id=f"{doc_id}:c{j}",
                        text=ch,
                    )
                )
        return all_chunks

    raise RuntimeError(
        f"No hay PDFs en {raw_dir} ni TXT en {processed_dir}. "
        f"Mete documentos en data/raw/ o data/processed/."
    )



def _save_chunks(chunks: List[Chunk], path: Path) -> None:
    payload = [
        {"doc_id": c.doc_id, "page": c.page, "chunk_id": c.chunk_id, "text": c.text}
        for c in chunks
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_chunks(path: Path) -> List[Chunk]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**d) for d in payload]


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Usamos cosine similarity:
      - normalizamos embeddings
      - IndexFlatIP (inner product)
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def _maybe_build_artifacts() -> None:
    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        return

    chunks = _build_chunks_from_data_dir(DATA_DIR)

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c.text for c in chunks]
    embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    index = _build_faiss_index(embs)

    faiss.write_index(index, str(INDEX_PATH))
    _save_chunks(chunks, CHUNKS_PATH)


@lru_cache(maxsize=1)
def _load_resources():
    """
    Carga (y cachea) índice + chunks + modelos de traducción + BART.
    Se ejecuta una sola vez por proceso (perfecto para Streamlit).
    """
    _maybe_build_artifacts()

    index = faiss.read_index(str(INDEX_PATH))
    chunks = _load_chunks(CHUNKS_PATH)

    # Embeddings (solo para query)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # Traducción
    es_en_tok = MarianTokenizer.from_pretrained(ES_EN_MODEL_NAME)
    es_en_model = MarianMTModel.from_pretrained(ES_EN_MODEL_NAME)

    en_es_tok = MarianTokenizer.from_pretrained(EN_ES_MODEL_NAME)
    en_es_model = MarianMTModel.from_pretrained(EN_ES_MODEL_NAME)

    # BART
    bart_tok = BartTokenizer.from_pretrained(BART_MODEL_NAME)
    bart_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME)

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    bart_model.to(device)
    es_en_model.to(device)
    en_es_model.to(device)

    return {
        "index": index,
        "chunks": chunks,
        "embedder": embedder,
        "es_en_tok": es_en_tok,
        "es_en_model": es_en_model,
        "en_es_tok": en_es_tok,
        "en_es_model": en_es_model,
        "bart_tok": bart_tok,
        "bart_model": bart_model,
        "device": device,
    }


# -----------------------------
# RAG steps (retrieve + generate)
# -----------------------------
def _translate_es_to_en(text: str, r) -> str:
    tok = r["es_en_tok"]
    model = r["es_en_model"]
    device = r["device"]
    inputs = tok([text], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=1024)
    return tok.decode(out[0], skip_special_tokens=True)


def _translate_en_to_es(text: str, r) -> str:
    tok = r["en_es_tok"]
    model = r["en_es_model"]
    device = r["device"]
    inputs = tok([text], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=512)
    return tok.decode(out[0], skip_special_tokens=True)


def _translate_es_to_en_chunks(text: str, r, chunk_chars: int = 500) -> str:
    parts = [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    translated = [_translate_es_to_en(p, r) for p in parts if p.strip()]
    return " ".join(translated)


def _summarize_bart(text_en: str, r, max_input_tokens: int = 1024, max_output_tokens: int = 220) -> str:
    tok = r["bart_tok"]
    model = r["bart_model"]
    device = r["device"]

    inputs = tok.encode(
        text_en,
        return_tensors="pt",
        max_length=max_input_tokens,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs,
            max_length=max_output_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return tok.decode(summary_ids[0], skip_special_tokens=True)


def _retrieve(query: str, r, top_k: int = 5) -> List[Dict[str, Any]]:
    embedder = r["embedder"]
    index = r["index"]
    chunks: List[Chunk] = r["chunks"]

    q = query.strip()
    q_vec = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    scores, idxs = index.search(q_vec, top_k)  # scores: (1,k), idxs:(1,k)
    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        c = chunks[int(idx)]
        results.append(
            {
                "doc_id": c.doc_id,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "score": float(score),
                "text": c.text,
            }
        )
    return results


def rag_query(question: str, k: int = 5) -> Dict[str, Any]:
    """
    Devuelve un dict estable para tu Streamlit:

    {
      "answer": str,
      "rejected": bool,
      "sources": [
         {"doc_id": str, "page": int|None, "chunk_id": str, "score": float, "snippet": str}
      ]
    }

    Requisito: si no hay doc relevante, NO responder (rejected=True). :contentReference[oaicite:1]{index=1}
    """
    r = _load_resources()

    q = (question or "").strip()
    if not q:
        return {"answer": "Escribe una pregunta.", "rejected": True, "sources": []}

    retrieved = _retrieve(q, r, top_k=k)
    if not retrieved:
        return {"answer": "No tengo información suficiente en los documentos.", "rejected": True, "sources": []}

    best_score = retrieved[0]["score"]
    if best_score < MIN_SCORE_TO_ANSWER:
        return {"answer": "No tengo información suficiente en los documentos.", "rejected": True, "sources": []}

    # Generación (igual que el notebook: ES->EN -> BART -> EN->ES)
    context_es = " ".join([x["text"] for x in retrieved])
    context_en = _translate_es_to_en_chunks(context_es, r)
    summary_en = _summarize_bart(context_en, r)
    answer_es = _translate_en_to_es(summary_en, r)

    sources = []
    for x in retrieved:
        sources.append(
            {
                "doc_id": x["doc_id"],
                "page": x["page"],
                "chunk_id": x["chunk_id"],
                "score": x["score"],
                "snippet": x["text"][:350] + ("..." if len(x["text"]) > 350 else ""),
            }
        )

    return {"answer": answer_es, "rejected": False, "sources": sources}
