# app/multi_classification_RAG_modal/jsonl_handler.py

import json
import hashlib
import time
from typing import List, Dict, Any
from fastapi import HTTPException
from app.multi_classification_RAG_modal.rag import get_vector_store, _chunk_text
from app.multi_classification_RAG_modal.embeddings import embed_texts
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONLProcessor:
    def __init__(self):
        self.vector_store = get_vector_store()

    def parse_jsonl_file(self, file_content: bytes) -> List[Dict[str, Any]]:
        """
        Parse JSONL file content and return list of JSON objects.
        """
        try:
            content = file_content.decode('utf-8')
            json_objects = []
            for line_num, line in enumerate(content.strip().split('\n'), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_obj = json.loads(line)
                    json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid JSON on line {line_num}: {str(e)}")
            if not json_objects:
                raise HTTPException(status_code=400, detail="No valid JSON objects found in file")
            return json_objects
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse JSONL file: {str(e)}")

    def extract_text_content(self, json_obj: Dict[str, Any]) -> str:
        """
        Extract text content from JSON object for embedding.
        """
        text = json_obj.get('text', '').strip()
        if text:
            return text
        return ""

    def create_metadata(self, json_obj: Dict[str, Any], filename: str, chunk_index: int, source: str = "text") -> Dict[str, Any]:
        """
        Create metadata with category from title. Updated for bilingual support.
        """
        return {
            "filename": filename,
            "source": source,
            "created_at": int(time.time()),
            "chunk_index": chunk_index,
            "total_chunks": 1,  # Updated later if chunking occurs
            "category": json_obj.get("title", ""),
            "section": json_obj.get("section", ""),
            "id": json_obj.get("id", ""),
            "language": "hi" if any(ord(c) > 127 for c in json_obj.get("text", "")) else "en"
        }

    def process_jsonl_batch(self, json_objects: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
        """
        Process a batch of JSON objects and upsert all vectors in one go.
        """
        total_objects = len(json_objects)
        successful_stores = 0
        failed_stores = 0
        total_chunks = 0
        images = []

        # Collect all chunks and metadata
        all_chunks = []
        all_metadatas = []
        for obj_index, json_obj in enumerate(json_objects):
            try:
                text_content = self.extract_text_content(json_obj)
                if text_content:
                    text_bytes = len(text_content.encode('utf-8'))
                    logger.info(f"JSON object {obj_index} text size: {text_bytes} bytes")
                    if text_bytes > 10000:  # Match rag.py threshold
                        chunks = _chunk_text(text_content, chunk_size=8000)
                        logger.info(f"JSON object {obj_index} split into {len(chunks)} chunks")
                    else:
                        chunks = [text_content]
                        logger.info(f"JSON object {obj_index} no chunking needed")
                    
                    for chunk_index, chunk in enumerate(chunks):
                        metadata = self.create_metadata(json_obj, filename, chunk_index, "text")
                        metadata["text"] = chunk[:500] + "..." if len(chunk) > 500 else chunk
                        metadata["total_chunks"] = len(chunks)
                        all_chunks.append(chunk)
                        all_metadatas.append(metadata)
                        total_chunks += 1
            except Exception as e:
                logger.error(f"Error processing JSON object {obj_index + 1}: {str(e)}")
                failed_stores += 1

        # Batch embed and upsert
        try:
            if all_chunks:
                logger.info(f"Embedding {len(all_chunks)} chunks for {filename}")
                embeddings = embed_texts(all_chunks, task="retrieval_document")
                vectors = []
                for i, (chunk, embedding, metadata) in enumerate(zip(all_chunks, embeddings, all_metadatas)):
                    if embedding:
                        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                        vector_id = f"doc_{chunk_hash}_{int(time.time())}_{i}"
                        vectors.append((vector_id, embedding, metadata))
                    else:
                        logger.warning(f"No embedding generated for chunk {i} of {filename}")
                        failed_stores += 1

                if vectors:
                    store = get_vector_store()
                    store.upsert(vectors=vectors)
                    successful_stores = len(vectors)
                    logger.info(f"Successfully upserted {successful_stores} vectors for {filename} in one batch")
                else:
                    logger.warning(f"No vectors to upsert for {filename}")
                    failed_stores += len(all_chunks)
            else:
                logger.warning(f"No chunks to process for {filename}")
                failed_stores += total_objects
        except Exception as e:
            logger.error(f"Error upserting vectors for {filename}: {str(e)}")
            failed_stores += len(all_chunks)

        return {
            "total_objects": total_objects,
            "successful_stores": successful_stores,
            "failed_stores": failed_stores,
            "total_chunks": total_chunks,
            "images": images,
            "image_count": len(images),
            "success_rate": (successful_stores / (successful_stores + failed_stores) * 100) if (successful_stores + failed_stores) > 0 else 0
        }

    def process_jsonl_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Complete JSONL file processing pipeline.
        """
        try:
            json_objects = self.parse_jsonl_file(file_content)
            results = self.process_jsonl_batch(json_objects, filename)
            results.update({
                "filename": filename,
                "file_size_bytes": len(file_content),
                "processing_time": time.time()
            })
            return results
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process JSONL file: {str(e)}")

def get_jsonl_processor() -> JSONLProcessor:
    """Get or create JSONL processor instance."""
    return JSONLProcessor()