# File: app/multimodal_chatbot/embeddings.py
"""
Google Gemini embeddings wrapper for text and image descriptions.

Usage:
    from embeddings import embed_text, embed_texts

    v = embed_text("hello world")
    vs = embed_texts(["doc one", "doc two"], task="retrieval_document")
"""

from typing import List, Sequence, Optional
from app.config import Settings
import google.generativeai as genai

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

    Args:
        text: Input string (e.g., PDF text or VLM-generated image description).
        task: One of {'retrieval_query', 'retrieval_document', 'semantic_similarity',
                      'classification', 'clustering', 'unspecified'}.
        api_key: Overrides GOOGLE_API_KEY env var.
        model: Gemini embedding model (default text-embedding-004).

    Returns:
        List[float]: Embedding vector.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("`text` must be a non-empty string.")
    if task not in _VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Valid: {_VALID_TASKS}")

    _configure(api_key)
    try:
        if len(text.encode('utf-8')) > 35000:
            text = text[:30000]  # Truncate to fit Gemini limits
        resp = genai.embed_content(
            model=model,
            content=text,
            task_type=task,
        )
        return list(resp["embedding"])
    except Exception as e:
        print(f"Error embedding text '{text[:100]}...': {e}")
        return []

def embed_texts(
    texts: Sequence[str],
    *,
    task: str = "retrieval_document",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> List[List[float]]:
    """
    Embed multiple strings (e.g., text chunks or image descriptions).

    Args:
        texts: Sequence of strings.
        task: Embedding task type.
        api_key: Overrides GOOGLE_API_KEY env var.
        model: Gemini embedding model.
        batch_size: Chunk size to avoid large requests.

    Returns:
        List[List[float]]: List of embedding vectors.
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