from __future__ import annotations
from typing import List, Sequence, Optional
from app.config import Settings
import os

# Lazy import
try:
    import google.generativeai as genai
except ImportError as e:
    raise ImportError(
        "Missing dependency 'google-generativeai'. Install with: pip install google-generativeai aiohttp"
    ) from e

# Config
DEFAULT_MODEL = "models/text-embedding-004"  # 768-dim
_VALID_TASKS = {
    "unspecified",
    "retrieval_query",
    "retrieval_document",
    "semantic_similarity",
    "classification",
    "clustering",
}

def _configure(api_key: Optional[str] = None) -> None:
    key = api_key or Settings.GOOGLE_API_KEY
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set. Export it or pass api_key=...")
    genai.configure(api_key=key)

def embed_text(
    text: str,
    *,
    task: str = "retrieval_query",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> List[float]:
    """
    Embed a single string and return its vector.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("`text` must be a non-empty string.")
    if task not in _VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Valid: {_VALID_TASKS}")

    _configure(api_key)
    resp = genai.embed_content(
        model=model,
        content=text,
        task_type=task,
    )
    return list(resp["embedding"])

def embed_texts(
    texts: Sequence[str],
    *,
    task: str = "retrieval_document",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> List[List[float]]:
    """
    Embed multiple strings. Returns vectors in the same order.
    """
    if not texts:
        return []
    if any((not isinstance(t, str) or not t.strip()) for t in texts):
        raise ValueError("All `texts` must be non-empty strings.")
    if task not in _VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Valid: {_VALID_TASKS}")

    _configure(api_key)

    out: List[List[float]] = []
    for text in texts:
        try:
            if len(text.encode('utf-8')) > 35000:
                print(f"Warning: Text too large ({len(text.encode('utf-8'))} bytes), truncating...")
                text = text[:30000]
            resp = genai.embed_content(
                model=model,
                content=text,
                task_type=task,
            )
            embedding = resp.get("embedding")
            if not embedding:
                print(f"Warning: No embedding returned for text: {text[:100]}...")
                continue
            out.append(list(embedding))
        except Exception as e:
            print(f"Error embedding text '{text[:100]}...': {e}")
            continue

    return out