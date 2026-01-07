from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from functools import lru_cache

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
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

# Modelos
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # BGE Multilingüe
BART_MODEL_NAME = "facebook/bart-large-cnn"
ES_EN_MODEL_NAME = "Helsinki-NLP/opus-mt-es-en"
EN_ES_MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"

MIN_RERANK_SCORE = -5.0 

# -----------------------------
# Types & Extraction
# -----------------------------
@dataclass
class Chunk:
    doc_id: str
    page: Optional[int]
    chunk_id: str
    text: str

def _extract_pdf_text_per_page(pdf_path: Path) -> List[Tuple[int, str]]:
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
        raise RuntimeError(f"Error leyendo PDF {pdf_path.name}: {e}")

def _clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return chunks

def _build_chunks_from_data_dir(data_dir: Path) -> List[Chunk]:
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    pdfs = sorted(list(raw_dir.glob("*.pdf"))) if raw_dir.exists() else []
    
    if pdfs:
        all_chunks: List[Chunk] = []
        for pdf in pdfs:
            doc_id = pdf.name
            pages = _extract_pdf_text_per_page(pdf)
            for page_num, page_text in pages:
                page_text = _clean_text(page_text)
                for j, ch in enumerate(_chunk_text(page_text)):
                    all_chunks.append(Chunk(doc_id=doc_id, page=page_num, chunk_id=f"{doc_id}:p{page_num}:c{j}", text=ch))
        return all_chunks

    txts = sorted(list(processed_dir.glob("*.txt"))) if processed_dir.exists() else []
    if txts:
        all_chunks: List[Chunk] = []
        for txt in txts:
            doc_id = txt.name
            text = _clean_text(txt.read_text(encoding="utf-8", errors="ignore"))
            for j, ch in enumerate(_chunk_text(text)):
                all_chunks.append(Chunk(doc_id=doc_id, page=None, chunk_id=f"{doc_id}:c{j}", text=ch))
        return all_chunks
    return []

def _save_chunks(chunks: List[Chunk], path: Path) -> None:
    payload = [{"doc_id": c.doc_id, "page": c.page, "chunk_id": c.chunk_id, "text": c.text} for c in chunks]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_chunks(path: Path) -> List[Chunk]:
    if not path.exists(): return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**d) for d in payload]

def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index

def _maybe_build_artifacts() -> None:
    if INDEX_PATH.exists() and CHUNKS_PATH.exists(): return
    chunks = _build_chunks_from_data_dir(DATA_DIR)
    if not chunks: return 
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c.text for c in chunks]
    embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = _build_faiss_index(embs)
    faiss.write_index(index, str(INDEX_PATH))
    _save_chunks(chunks, CHUNKS_PATH)

@lru_cache(maxsize=1)
def _load_resources():
    _maybe_build_artifacts()
    if not INDEX_PATH.exists(): raise RuntimeError("No se pudo crear el índice.")
    
    # --- CARGA DE DATOS ---
    index = faiss.read_index(str(INDEX_PATH))
    chunks = _load_chunks(CHUNKS_PATH)
    
    # --- DETECCIÓN DE HARDWARE (AQUÍ ESTABA EL PROBLEMA) ---
    if torch.cuda.is_available():
        device = "cuda"  # ¡Para tu RTX 5070 Ti!
        print(f"🚀 USANDO GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"   # Para Mac
    else:
        device = "cpu"   # Freno de mano
        print("⚠️ USANDO CPU (Lento)")

    # --- CARGA DE MODELOS ---
    # Embedder y Reranker (Suelen gestionarse solos, pero forzamos coherencia)
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    reranker = CrossEncoder(RERANK_MODEL_NAME, max_length=512, device=device)
    
    # Traducción
    es_en_tok = MarianTokenizer.from_pretrained(ES_EN_MODEL_NAME)
    es_en_model = MarianMTModel.from_pretrained(ES_EN_MODEL_NAME).to(device)
    
    en_es_tok = MarianTokenizer.from_pretrained(EN_ES_MODEL_NAME)
    en_es_model = MarianMTModel.from_pretrained(EN_ES_MODEL_NAME).to(device)
    
    # BART (Resumen)
    bart_tok = BartTokenizer.from_pretrained(BART_MODEL_NAME)
    bart_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME).to(device)
    
    return {
        "index": index, "chunks": chunks, 
        "embedder": embedder, "reranker": reranker,
        "es_en_tok": es_en_tok, "es_en_model": es_en_model,
        "en_es_tok": en_es_tok, "en_es_model": en_es_model,
        "bart_tok": bart_tok, "bart_model": bart_model, 
        "device": device
    }

# -----------------------------
# Traducción (CORREGIDA: max_length=512)
# -----------------------------
def _translate_es_to_en(text: str, r) -> str:
    tok, model, device = r["es_en_tok"], r["es_en_model"], r["device"]
    inputs = tok([text], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=512)
    return tok.decode(out[0], skip_special_tokens=True)

def _translate_en_to_es(text: str, r) -> str:
    tok, model, device = r["en_es_tok"], r["en_es_model"], r["device"]
    inputs = tok([text], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=512)
    return tok.decode(out[0], skip_special_tokens=True)

def _translate_es_to_en_chunks(text: str, r, chunk_chars: int = 500) -> str:
    parts = [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    translated = [_translate_es_to_en(p, r) for p in parts if p.strip()]
    return " ".join(translated)

def _summarize_bart(text_en: str, r, max_input_tokens: int = 1024, max_output_tokens: int = 220) -> str:
    tok, model, device = r["bart_tok"], r["bart_model"], r["device"]
    inputs = tok.encode(text_en, return_tensors="pt", max_length=max_input_tokens, truncation=True).to(device)
    with torch.no_grad():
        summary_ids = model.generate(inputs, max_length=max_output_tokens, num_beams=4, early_stopping=True)
    return tok.decode(summary_ids[0], skip_special_tokens=True)

# -----------------------------
# Retrieval + Re-ranking
# -----------------------------
def _retrieve(query: str, r, top_k: int = 5) -> List[Dict[str, Any]]:
    # 1. Recuperación amplia (Bi-Encoder)
    candidates_k = top_k * 4 # Pedimos más para luego filtrar duplicados
    q_vec = r["embedder"].encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, idxs = r["index"].search(q_vec, candidates_k)
    
    initial_results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0: continue
        initial_results.append(r["chunks"][int(idx)])
    if not initial_results: return []

    # 2. Re-ranking (Cross-Encoder)
    pairs = [[query, doc.text] for doc in initial_results]
    cross_scores = r["reranker"].predict(pairs)

    reranked = []
    for doc, score in zip(initial_results, cross_scores):
        reranked.append({
            "doc_id": doc.doc_id, "page": doc.page, "chunk_id": doc.chunk_id, 
            "score": float(score), "text": doc.text
        })
    # Ordenar por score
    reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
    
    return reranked

# -----------------------------
# Main Query
# -----------------------------
def rag_query(question: str, k: int = 5, target_lang: str = "es") -> Dict[str, Any]:
    r = _load_resources()
    q = (question or "").strip()
    
    msg_no_info = "No tengo información suficiente en los documentos." if target_lang == "es" else "Insufficient information in documents."
    
    if not q: return {"answer": "...", "rejected": True, "sources": []}

    retrieved = _retrieve(q, r, top_k=k) # Trae muchos candidatos
    
    if not retrieved or retrieved[0]["score"] < MIN_RERANK_SCORE:
        return {"answer": msg_no_info, "rejected": True, "sources": []}

    # --- DEDUPLICACIÓN (SOLUCIÓN A TU PROBLEMA) ---
    # Filtramos para no repetir (Doc, Página). Nos quedamos con el primero que aparece (el de mayor score).
    unique_sources = []
    seen_pages = set()
    
    for item in retrieved:
        # Clave única: Nombre del PDF + Número de página
        key = (item["doc_id"], item["page"])
        if key not in seen_pages:
            unique_sources.append(item)
            seen_pages.add(key)
        
        # Paramos cuando tengamos los 'k' que pidió el usuario
        if len(unique_sources) >= k:
            break
            
    # Usamos los documentos únicos para generar la respuesta
    context_es = " ".join([x["text"] for x in unique_sources])
    context_en = _translate_es_to_en_chunks(context_es, r)
    summary_en = _summarize_bart(context_en, r)
    final_answer = _translate_en_to_es(summary_en, r) if target_lang == "es" else summary_en

    sources = []
    for x in unique_sources:
        sources.append({
            "doc_id": x["doc_id"], "page": x["page"], 
            "score": x["score"], "snippet": x["text"][:300] + "..."
        })

    return {"answer": final_answer, "rejected": False, "sources": sources}