# app/multi_classification_RAG_modal/rag_faiss.py
import json
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 300  # Approx tokens per chunk
CHUNK_OVERLAP = 50
INDEX_PATH = Path("faiss_index.faiss")  # Saved in project root
META_PATH = Path("faiss_meta.pkl")  # Saved in project root

# Helper: Chunk text
def _chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# FAISS class
class FAISSRetriever:
    _model = None
    _index = None
    _meta = None  # List[Dict] parallel to vectors

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer(MODEL_NAME)
        return cls._model

    @classmethod
    def build_index(cls, jsonl_path: str, force_rebuild: bool = False):
        if not force_rebuild and INDEX_PATH.exists() and META_PATH.exists():
            cls.load_index()
            logger.info("Loaded existing FAISS index.")
            return

        model = cls._load_model()
        vectors = []
        meta = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Line {line_no} invalid JSON – skipping")
                    continue

                text = obj.get("text", "")
                if not text:
                    continue

                chunks = _chunk_text(text)
                for idx, chunk in enumerate(chunks):
                    vec = model.encode(chunk, normalize_embeddings=True)
                    vectors.append(vec)

                    meta.append({
                        "id": obj.get("id", f"doc_{line_no}"),
                        "text": chunk,
                        "title": obj.get("title", ""),
                        "section": obj.get("section", ""),
                        "url": obj.get("url", ""),
                        "metadata": obj.get("metadata", {}),
                        "chunk_index": idx,
                        "original_line": line_no
                    })

        if not vectors:
            logger.error("No vectors generated – JSONL empty?")
            return

        # FAISS index (FlatIP for small datasets)
        dim = len(vectors[0])
        index = faiss.IndexFlatIP(dim)
        index.add(np.array(vectors, dtype=np.float32))

        cls._index = index
        cls._meta = meta

        # Save
        faiss.write_index(index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump(meta, f)

        logger.info(f"Built FAISS: {index.ntotal} vectors, dim={dim}")

    @classmethod
    def load_index(cls):
        if cls._index is not None:
            return
        cls._index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            cls._meta = pickle.load(f)
        logger.info(f"Loaded FAISS: {cls._index.ntotal} vectors")

    @classmethod
    def retrieve(cls, query: str, top_k: int = 5) -> List[Dict]:
        if cls._index is None:
            cls.load_index()

        model = cls._load_model()
        q_vec = model.encode(query, normalize_embeddings=True).astype(np.float32)[np.newaxis]

        scores, indices = cls._index.search(q_vec, top_k)
        results = []

        for i in range(top_k):
            idx = indices[0, i]
            if idx == -1:
                break
            results.append({
                "score": float(scores[0, i]),
                "metadata": cls._meta[idx]
            })

        return results