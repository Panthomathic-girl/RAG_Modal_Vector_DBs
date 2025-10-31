# app/multi_classification_RAG_modal/embeddings.py
# Gemini multilingual embeddings (Hindi + English) with FULL DEBUG LOGGING

from __future__ import annotations
from typing import List, Sequence, Optional
from app.config import Settings
import logging
import time

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import
try:
    import google.generativeai as genai
except ImportError as e:
    raise ImportError(
        "Missing dependency 'google-generativeai'. Install with: pip install google-generativeai"
    ) from e

# Config
DEFAULT_MODEL = "models/text-embedding-004"  # 768-dim
BATCH_SIZE = 20  # ← Gemini supports up to 20 texts per request
_VALID_TASKS = {
    "unspecified",
    "retrieval_query",
    "retrieval_document",
    "semantic_similarity",
    "classification",
    "clustering",
}

def _configure(api_key: Optional[str] = None) -> None:
    key = api_key or Settings().GOOGLE_API_KEY
    if not key:
        logger.error("GOOGLE_API_KEY is missing! Set in .env or pass api_key=")
        raise RuntimeError("GOOGLE_API_KEY not set. Export it or pass api_key=...")
    
    logger.info(f"Configuring Gemini with API key: {key[:10]}...{key[-4:]}")
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
    FULL DEBUG OUTPUT.
    """
    if not isinstance(text, str) or not text.strip():
        logger.error("Invalid input: `text` must be a non-empty string.")
        raise ValueError("`text` must be a non-empty string.")
    if task not in _VALID_TASKS:
        logger.error(f"Invalid task '{task}'. Valid: {_VALID_TASKS}")
        raise ValueError(f"Invalid task '{task}'. Valid: {_VALID_TASKS}")

    logger.info(f"Embedding text (task={task}, model={model})")
    logger.debug(f"Text preview: '{text[:200]}{'...' if len(text) > 200 else ''}'")

    _configure(api_key)

    try:
        resp = genai.embed_content(
            model=model,
            content=text,
            task_type=task,
        )
        embedding = resp.get("embedding")
        if not embedding:
            logger.warning("Gemini returned empty embedding!")
            return []
        
        logger.info(f"Success: Embedding generated (dim={len(embedding)})")
        return list(embedding)

    except Exception as e:
        logger.error(f"Gemini embedding FAILED: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []


def embed_texts(
    texts: Sequence[str],
    *,
    task: str = "retrieval_document",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> List[List[float]]:
    """
    Embed multiple strings using BATCHED API calls (20 at a time).
    Returns vectors in the same order.
    FULL DEBUG + PROGRESS TRACKING.
    """
    if not texts:
        logger.info("No texts to embed.")
        return []
    if any((not isinstance(t, str) or not t.strip()) for t in texts):
        logger.error("All `texts` must be non-empty strings.")
        raise ValueError("All `texts` must be non-empty strings.")
    if task not in _VALID_TASKS:
        logger.error(f"Invalid task '{task}'. Valid: {_VALID_TASKS}")
        raise ValueError(f"Invalid task '{task}'. Valid: {_VALID_TASKS}")

    logger.info(f"Starting BATCHED embedding: {len(texts)} texts (task={task}, model={model})")
    _configure(api_key)

    out: List[List[float]] = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, i in enumerate(range(0, len(texts), BATCH_SIZE), 1):
        batch = texts[i:i + BATCH_SIZE]
        batch = [t[:30000] for t in batch]  # Truncate to ~30K chars
        logger.info(f"[{batch_idx}/{total_batches}] Embedding batch of {len(batch)} chunks...")

        try:
            resp = genai.embed_content(
                model=model,
                content=batch,
                task_type=task,
            )
            embeddings = resp.get("embedding", [])
            
            # Handle mismatch
            if len(embeddings) != len(batch):
                logger.warning(f"Expected {len(batch)}, got {len(embeddings)} embeddings. Padding...")
                embeddings += [[]] * (len(batch) - len(embeddings))

            for idx, embedding in enumerate(embeddings):
                if embedding:
                    embedding_list = list(embedding)
                    logger.info(f"  [Batch {batch_idx}, #{idx+1}] Success: Got embedding (dim={len(embedding_list)})")
                    out.append(embedding_list)
                else:
                    logger.warning(f"  [Batch {batch_idx}, #{idx+1}] Empty embedding")
                    out.append([])

            time.sleep(0.15)  # Prevent rate limiting

        except Exception as e:
            logger.error(f"[Batch {batch_idx}] Embedding FAILED: {type(e).__name__}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            out.extend([[] for _ in batch])

    success_count = sum(1 for emb in out if len(emb) == 768)
    logger.info(f"BATCHED embedding complete: {success_count}/{len(texts)} successful")
    return out