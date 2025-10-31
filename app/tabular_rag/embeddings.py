from typing import List
from app.config import Settings  # Updated import
import google.generativeai as genai
import time

genai.configure(api_key=Settings.GOOGLE_API_KEY)

def embed_text(text: str, task: str = "retrieval_query", max_retries: int = 3) -> List[float]:
    """Embed a single text string with retry logic."""
    for attempt in range(max_retries):
        try:
            response = genai.embed_content(model="models/text-embedding-004", content=text, task_type=task)
            return response["embedding"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            print(f"Embedding error after {max_retries} attempts: {e}")
            return []

def embed_texts(texts: list, task: str = "retrieval_document") -> List[List[float]]:
    """Embed multiple texts."""
    return [embed_text(text, task) for text in texts]