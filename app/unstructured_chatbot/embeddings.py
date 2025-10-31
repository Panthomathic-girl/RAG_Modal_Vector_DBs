# File: app/unstructured_chatbot/embeddings.py
"""
Lightweight Gemini embeddings wrapper.

Usage:
    from embeddings_gemini import embed_text, embed_texts

    v = embed_text("hello world")
    vs = embed_texts(["doc one", "doc two"], task="retrieval_document")
"""

from __future__ import annotations
from typing import List, Sequence, Optional
from app.config import Settings
import os

# Lazy import so your app can still start without the package installed
try:
    import google.generativeai as genai
except ImportError as e:
    raise ImportError(
        "Missing dependency 'google-generativeai'. "
        "Install with: pip install google-generativeai aiohttp"
    ) from e


# ---- Config ----
DEFAULT_MODEL = "models/text-embedding-004"  # 768-dim as of current API
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
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Export it or pass api_key=..."
        )
    genai.configure(api_key=key)


# ---- Public API ----
def embed_text(
    text: str,
    *,
    task: str = "retrieval_query",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> List[float]:
    """
    Embed a single string and return its vector.

    Args:
        text: input string
        task: one of {'retrieval_query','retrieval_document','semantic_similarity',
                      'classification','clustering','unspecified'}
        api_key: overrides GOOGLE_API_KEY env var
        model: Gemini embedding model (default text-embedding-004)

    Returns:
        List[float]: embedding vector
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
    # API returns {"embedding": [floats]}
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
    Embed multiple strings. Returns vectors in the SAME order.

    Args:
        texts: sequence of strings
        task: embedding task type (see `embed_text`)
        api_key: overrides GOOGLE_API_KEY env var
        model: Gemini embedding model
        batch_size: chunk size to avoid very large requests

    Returns:
        List[List[float]]
    """
    if not texts:
        return []
    if any((not isinstance(t, str) or not t.strip()) for t in texts):
        raise ValueError("All `texts` must be non-empty strings.")
    if task not in _VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Valid: {_VALID_TASKS}")

    _configure(api_key)

    out: List[List[float]] = []
    # Process texts one by one to avoid batch issues
    for text in texts:
        try:
            # Check text size limit (Gemini has ~36KB limit)
            if len(text.encode('utf-8')) > 35000:  # Leave some buffer
                print(f"Warning: Text too large ({len(text.encode('utf-8'))} bytes), truncating...")
                # Truncate text to fit within limits
                text = text[:30000]  # Approximate character limit
            
            resp = genai.embed_content(
                model=model,
                content=text,
                task_type=task,
            )
            # Single text returns {"embedding": [...]}
            embedding = resp.get("embedding")
            if not embedding:
                print(f"Warning: No embedding returned for text: {text[:100]}...")
                # Skip this text instead of creating zero vector
                continue
            out.append(list(embedding))
        except Exception as e:
            print(f"Error embedding text '{text[:100]}...': {e}")
            # Skip this text instead of creating zero vector
            continue

    return out