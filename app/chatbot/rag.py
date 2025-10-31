# vectorstore_pinecone.py
"""
Pinecone helper module with Google Gemini embeddings.

Capabilities:
- ensure_index / create serverless index (dimension/metric configurable)
- upsert_texts() -> embeds with Google Gemini and writes to Pinecone
- query_by_text() -> search using text queries
- delete_by_filter() -> delete vectors by metadata filter
- describe_stats() -> get index statistics

Environment:
    PINECONE_API_KEY   (required)
    GOOGLE_API_KEY     (required for embeddings)
    PINECONE_INDEX     (default index name if not provided to class)
    PINECONE_CLOUD     (default "aws")
    PINECONE_REGION    (default "us-east-1")

Requirements:
    pinecone>=5.0.0
    google-generativeai>=0.3.0
    aiohttp>=3.9.0

Embedding Model:
    Provider: Google Gemini
    Model: text-embedding-004
    Dimensions: 768
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Sequence
from app.config import Settings
from pinecone import Pinecone, ServerlessSpec
from app.chatbot.embeddings import embed_texts, embed_text



DEFAULT_METRIC = "cosine"  # cosine | dotproduct | euclidean
DEFAULT_CLOUD = Settings.PINECONE_CLOUD
DEFAULT_REGION = Settings.PINECONE_REGION
DEFAULT_INDEX = Settings.PINECONE_INDEX


class PineconeVectorStore:
    """
    Thin wrapper around Pinecone Index for typical vector DB operations.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        cloud: str = DEFAULT_CLOUD,
        region: str = DEFAULT_REGION,
        create_if_missing: bool = True,
        dimension: Optional[int] = None,
        metric: str = DEFAULT_METRIC,
        pod_type: Optional[str] = None,  # not used for serverless, kept for forward-compat
        wait_until_ready: bool = True,
    ) -> None:
        """
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env)
            index_name: Pinecone index name (defaults to PINECONE_INDEX env)
            cloud, region: ServerlessSpec settings
            create_if_missing: create index if it doesn't exist
            dimension: required for new index creation (embedding size)
            metric: cosine | dotproduct | euclidean
            wait_until_ready: poll until index is ready after creation
        """
        self.api_key = api_key or Settings.PINECONE_API_KEY
        if not self.api_key:
            raise RuntimeError("Missing environment variable: PINECONE_API_KEY")
        self.index_name = index_name or DEFAULT_INDEX
        if not self.index_name:
            raise RuntimeError("Index name not provided. Set PINECONE_INDEX or pass index_name=...")

        self.cloud = cloud
        self.region = region
        self.metric = metric
        self.dimension = dimension
        self._pc = Pinecone(api_key=self.api_key)

        # Create or attach to index
        if self.index_name not in [ix["name"] for ix in self._pc.list_indexes()]:
            if not create_if_missing:
                raise RuntimeError(
                    f"Pinecone index '{self.index_name}' does not exist, and create_if_missing=False."
                )
            
            # Create index for external embeddings
            self._pc.create_index(
                name=self.index_name,
                dimension=768,  # Standard dimension for external embeddings
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud=self.cloud, 
                    region=self.region
                ),
            )
            if wait_until_ready:
                self._wait_for_index_ready(self.index_name)
        self.index = self._pc.Index(self.index_name)

    # -----------------------
    # Index utils
    # -----------------------
    def _wait_for_index_ready(self, name: str, timeout_sec: int = 300, poll: float = 2.0) -> None:
        start = time.time()
        while time.time() - start < timeout_sec:
            desc = self._pc.describe_index(name)
            if desc.get("status", {}).get("ready"):
                return
            time.sleep(poll)
        raise TimeoutError(f"Index '{name}' not ready after {timeout_sec} seconds.")


    def upsert_texts(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Optional[Sequence[Mapping[str, Any]]] = None,
        *,
        namespace: Optional[str] = None,
        batch_size: int = 100,
    ) -> None:
        """
        Upsert texts using either Pinecone's built-in embedding or external embeddings.

        Each position across (ids, texts, metadatas) corresponds.
        """
        if len(ids) != len(texts):
            raise ValueError("ids and texts must be the same length.")

        if metadatas and len(metadatas) != len(texts):
            raise ValueError("metadatas, if provided, must match texts length.")

        # Use Google embeddings
        try:
            vectors = embed_texts(texts)
            
            # Prepare vectors with embeddings
            upsert_vectors = []
            for i, (text_id, text, metadata, embedding) in enumerate(zip(ids, texts, metadatas or [{}] * len(texts), vectors)):
                meta = dict(metadata or {})
                if "text" not in meta:
                    meta["text"] = text
                
                upsert_vectors.append({
                    "id": str(text_id),
                    "values": embedding,
                    "metadata": meta
                })
            
            # Upsert in batches
            for i in range(0, len(upsert_vectors), batch_size):
                batch = upsert_vectors[i : i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                print(f"Successfully upserted batch of {len(batch)} vectors with external embeddings")
                
        except ImportError:
            raise RuntimeError("Google embeddings not available. Please install google-generativeai.")


    def query_by_text(
        self,
        text: str,
        *,
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Mapping[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Query using external embeddings.
        """
        # Use Google embeddings
        try:
            embedding = embed_text(text)
            return self.index.query(
                vector=embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_values=include_values,
                include_metadata=include_metadata,
            )
        except ImportError:
            raise RuntimeError("Google embeddings not available. Please install google-generativeai.")


    def delete_by_filter(self, metadata_filter: Mapping[str, Any], *, namespace: Optional[str] = None) -> None:
        """
        Deletes vectors matching a metadata filter.
        Example filter: {"category": {"$eq": "blog"}}
        """
        self.index.delete(filter=dict(metadata_filter), namespace=namespace)


# Global instance for easy access
_vector_store = None


def get_vector_store() -> PineconeVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = PineconeVectorStore(
            create_if_missing=True
        )
    return _vector_store


def upsert_texts(text: str, filename: str = None, file_type: str = None) -> bool:
    """
    Simple function to upsert a single text to the vector database.
    Automatically chunks large texts to fit within embedding limits.
    
    Args:
        text: The text content to store
        filename: Optional filename for metadata
        file_type: Optional file type for metadata
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate input
        if not text or not text.strip():
            print("Warning: Empty text provided, skipping vector storage")
            return False
            
        store = get_vector_store()
        
        # Check if text is too large and needs chunking
        text_bytes = len(text.encode('utf-8'))
        if text_bytes > 50000:  # Pinecone can handle larger chunks
            print(f"Text is large ({text_bytes} bytes), chunking into smaller pieces...")
            chunks = _chunk_text(text, chunk_size=40000)  # 40KB chunks for Pinecone
            print(f"Split into {len(chunks)} chunks")
        else:
            chunks = [text]
        
        success_count = 0
        for i, chunk in enumerate(chunks):
            try:
                # Generate a unique ID for each chunk
                import hashlib
                import time
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                vector_id = f"doc_{chunk_hash}_{int(time.time())}_{i}"
                
                # Create metadata
                metadata = {
                    "filename": filename or "unknown",
                    "file_type": file_type or "unknown",
                    "text": chunk[:500] + "..." if len(chunk) > 500 else chunk,
                    "created_at": int(time.time()),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                print(f"Storing chunk {i+1}/{len(chunks)} in vector database: {filename} ({len(chunk)} chars)")
                
                # Upsert the chunk
                store.upsert_texts(
                    ids=[vector_id],
                    texts=[chunk],
                    metadatas=[metadata]
                )
                print(f"Successfully stored chunk with ID: {vector_id}")
                success_count += 1
                
            except Exception as chunk_error:
                print(f"Failed to store chunk {i+1}: {chunk_error}")
                continue
        
        if success_count > 0:
            print(f"Successfully stored {success_count}/{len(chunks)} chunks")
            return True
        else:
            print("Failed to store any chunks")
            return False
            
    except Exception as e:
        print(f"Error upserting text to vector database: {e}")
        import traceback
        traceback.print_exc()
        return False


def _chunk_text(text: str, chunk_size: int = 25000) -> list:
    """
    Split text into chunks of approximately the specified size.
    Tries to split at sentence boundaries when possible.
    """
    import re
    
    # First, try to split by paragraphs (double newlines)
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk size
        if len((current_chunk + paragraph).encode('utf-8')) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # Single paragraph is too large, split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len((current_chunk + sentence).encode('utf-8')) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            # Single sentence is too large, force split
                            chunks.append(sentence[:chunk_size])
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
