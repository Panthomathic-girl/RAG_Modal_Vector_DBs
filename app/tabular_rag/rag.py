import os
import time
import hashlib
import logging
from pinecone import Pinecone, ServerlessSpec
from app.config import Settings  # Updated import
from app.tabular_rag.embeddings import embed_text, embed_texts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_vector_store():
    """Initialize and return Pinecone vector store for tabular data."""
    logger.info("Initializing Pinecone vector store...")
    pc = Pinecone(api_key=Settings.PINECONE_API_KEY)
    index_name = Settings.PINECONE_TABULAR_INDEX
    if index_name not in pc.list_indexes().names():
        logger.info(f"Creating index {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=768,  # Matches Gemini embedding-004 dimension
            metric="cosine",
            spec=ServerlessSpec(cloud=Settings.PINECONE_CLOUD, region=Settings.PINECONE_REGION)
        )
        while not pc.describe_index(index_name).status['ready']:
            logger.info(f"Waiting for index {index_name} to be ready...")
            time.sleep(1)
    logger.info(f"Connected to index {index_name}")
    return pc.Index(index_name)

def upsert_texts(data: list, filename: str, source: str = "csv") -> int:
    """Upsert text data from CSV, SQL, or NoSQL into Pinecone in batches."""
    logger.info(f"Starting upsert for {len(data)} rows from {source} source, filename: {filename}")
    store = get_vector_store()
    total_stored = 0
    batch_size = 100  # Reduced batch size to prevent timeouts and memory issues

    for batch_start in range(0, len(data), batch_size):
        batch = data[batch_start:batch_start + batch_size]
        logger.info(f"Processing batch {batch_start // batch_size + 1} with {len(batch)} rows")
        
        # Convert rows to text strings
        texts = [" | ".join(f"{k}: {str(v)[:100]}" for k, v in row.items()) for row in batch]
        
        # Generate embeddings for the batch
        start_time = time.time()
        embeddings = embed_texts(texts, task="retrieval_document")
        logger.info(f"Embedding generation took {time.time() - start_time:.2f} seconds for {len(texts)} rows")
        
        # Prepare vectors for upsert
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            if embedding:
                vector_id = f"{source}_{hashlib.md5(text.encode()).hexdigest()[:8]}_{batch_start + i}_{int(time.time())}"
                metadata = {
                    "filename": filename if source == "csv" else f"{source}_data",
                    "source": source,
                    "text": text[:500],  # Truncated for efficiency
                    "created_at": int(time.time()),
                    "row_index": batch_start + i
                }
                vectors.append((vector_id, embedding, metadata))
        
        # Upsert the batch to Pinecone
        if vectors:
            start_time = time.time()
            store.upsert(vectors, namespace=source)
            total_stored += len(vectors)
            logger.info(f"Upserted {len(vectors)} vectors in {time.time() - start_time:.2f} seconds")
        else:
            logger.warning("No valid embeddings generated for this batch")
        
        # Small delay to prevent rate-limiting
        time.sleep(0.5)

    logger.info(f"Completed upsert: {total_stored} vectors stored")
    return total_stored

# rag.py → query_vector_store()
def query_vector_store(query: str, top_k: int = 5, filter: dict = None) -> dict:
    store = get_vector_store()
    embedding = embed_text(query, task="retrieval_query")
    if not embedding:
        return {"matches": []}
    
    # ← FIX: namespace is top-level
    namespace = filter.pop("source", None) if filter else None
    results = store.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter,
        namespace=namespace  # ← HERE
    )
    return results