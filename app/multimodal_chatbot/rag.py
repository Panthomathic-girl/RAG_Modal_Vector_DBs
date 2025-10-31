import os
import time
import re
import hashlib
from pinecone import Pinecone, ServerlessSpec
from app.config import Settings
from app.multimodal_chatbot.embeddings import embed_text, embed_texts

def get_vector_store():
    """
    Initialize and return Pinecone vector store using Pinecone class.
    """
    try:
        pc = Pinecone(api_key=Settings.PINECONE_API_KEY)
        index_name = Settings.PINECONE_UNSTRUCTURED_INDEX
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,  # Gemini text-embedding-004 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=Settings.PINECONE_CLOUD,
                    region=Settings.PINECONE_REGION
                )
            )
            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(2)
        return pc.Index(index_name)
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        raise

def _chunk_text(text: str, chunk_size: int = 40000) -> list:
    """
    Split text into chunks of approximately the specified size.
    Tries to split at paragraph boundaries when possible.
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len((current_chunk + paragraph).encode('utf-8')) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len((current_chunk + sentence).encode('utf-8')) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            chunks.append(sentence[:chunk_size])
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def upsert_texts(text: str, filename: str, file_type: str, source: str = "text", extra_metadata: dict = None) -> bool:
    """
    Split text into chunks, generate embeddings, and upsert to Pinecone.
    Args:
        text: Text to store (e.g., PDF text or VLM-generated image description).
        filename: Name of the source file.
        file_type: Type of the file (e.g., 'pdf').
        source: Source type ('text' or 'image').
        extra_metadata: Additional metadata (e.g., page_number for images).
    Returns:
        bool: True if successfully stored, False otherwise.
    """
    try:
        store = get_vector_store()
        text_bytes = len(text.encode('utf-8'))
        if text_bytes > 50000:
            print(f"Text is large ({text_bytes} bytes), chunking...")
            chunks = _chunk_text(text, chunk_size=40000)
            print(f"Split into {len(chunks)} chunks")
        else:
            chunks = [text]
        
        vectors = []
        embeddings = embed_texts(chunks, task="retrieval_document")
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding:
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                vector_id = f"doc_{chunk_hash}_{int(time.time())}_{i}"
                metadata = {
                    "filename": filename,
                    "file_type": file_type,
                    "text": chunk[:500] + "..." if len(chunk) > 500 else chunk,
                    "source": source, 
                    "created_at": int(time.time()),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                if extra_metadata:
                    metadata.update(extra_metadata)
                vectors.append((vector_id, embedding, metadata))
        
        if vectors:
            store.upsert(vectors=vectors)
            print(f"Successfully upserted {len(vectors)} vectors")
            return True
        else:
            print("No vectors to upsert")
            return False
    except Exception as e:
        print(f"Error upserting texts: {e}")
        return False

def query_vector_store(text: str, top_k: int = 5, filter: dict = None) -> dict:
    """
    Query Pinecone for similar texts.
    """
    try:
        store = get_vector_store()
        embedding = embed_text(text, task="retrieval_query")
        if not embedding:
            print(f"No embedding generated for query: {text}")
            return {"matches": []}
        results = store.query(vector=embedding, top_k=top_k, include_metadata=True, filter=filter)
        print(f"Query results: {results}")
        return results
    except Exception as e:
        print(f"Error querying vector store: {e}")
        return {"matches": []}