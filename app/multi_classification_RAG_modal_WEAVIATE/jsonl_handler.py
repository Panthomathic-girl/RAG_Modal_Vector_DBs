# app/multi_classification_RAG_modal_WEAVIATE/jsonl_handler.py

import json
import time
from typing import List, Dict, Any
from fastapi import HTTPException
from app.multi_classification_RAG_modal_WEAVIATE.rag_weaviate import get_vector_store, _chunk_text
from app.multi_classification_RAG_modal_WEAVIATE.embeddings import embed_texts
from weaviate.util import generate_uuid5
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONLProcessor:
    def __init__(self):
        self.client, self.collection = get_vector_store()
        logger.info("JSONLProcessor initialized with Weaviate collection")

    def parse_jsonl_file(self, file_content: bytes) -> List[Dict[str, Any]]:
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
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid JSON on line {line_num}: {str(e)}"
                    )
            if not json_objects:
                raise HTTPException(status_code=400, detail="No valid JSON objects found in file")
            return json_objects
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse JSONL file: {str(e)}")

    def extract_text_content(self, json_obj: Dict[str, Any]) -> str:
        text = json_obj.get('text', '').strip()
        return text if text else ""

    def create_metadata(
        self,
        json_obj: Dict[str, Any],
        filename: str,
        chunk_index: int,
        total_chunks: int,
        source: str = "text"
    ) -> Dict[str, Any]:
        full_text = json_obj.get("text", "")
        language = "hi" if any(ord(c) > 127 for c in full_text) else "en"

        return {
            "filename": filename,
            "source": source,
            "created_at": int(time.time()),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "category": json_obj.get("title", ""),
            "section": json_obj.get("section", ""),
            "source_id": json_obj.get("id", ""),
            "language": language,
            "text": ""  # Will be filled later
        }

    def process_jsonl_batch(self, json_objects: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
        total_objects = len(json_objects)
        successful_stores = 0
        failed_stores = 0
        total_chunks = 0
        images = []

        all_chunks = []
        all_metadatas = []

        # Step 1: Chunk all texts
        for obj_index, json_obj in enumerate(json_objects):
            try:
                text_content = self.extract_text_content(json_obj)
                if not text_content:
                    logger.warning(f"Empty text in object {obj_index}, skipping")
                    failed_stores += 1
                    continue

                text_bytes = len(text_content.encode('utf-8'))
                logger.info(f"JSON object {obj_index} text size: {text_bytes} bytes")

                if text_bytes > 10000:
                    chunks = _chunk_text(text_content, chunk_size=8000)
                    logger.info(f"JSON object {obj_index} split into {len(chunks)} chunks")
                else:
                    chunks = [text_content]
                    logger.info(f"JSON object {obj_index} no chunking needed")

                for chunk_index, chunk in enumerate(chunks):
                    metadata = self.create_metadata(
                        json_obj=json_obj,
                        filename=filename,
                        chunk_index=chunk_index,
                        total_chunks=len(chunks),
                        source="text"
                    )
                    metadata["text"] = chunk[:500] + ("..." if len(chunk) > 500 else "")
                    all_chunks.append(chunk)
                    all_metadatas.append(metadata)
                    total_chunks += 1

            except Exception as e:
                logger.error(f"Error processing JSON object {obj_index}: {str(e)}")
                failed_stores += 1

        # Step 2: Batch embed
        if all_chunks:
            logger.info(f"Embedding {len(all_chunks)} chunks for {filename}")
            embeddings = embed_texts(all_chunks, task="retrieval_document")

            # Step 3: gRPC-safe batch insert with dynamic()
            successful_count = 0
            failed_count = 0

            # USE dynamic() → returns BatchExecutor with failed_objects
            with self.collection.batch.dynamic() as batch:
                for i, (chunk, embedding, metadata) in enumerate(zip(all_chunks, embeddings, all_metadatas)):
                    if not embedding or len(embedding) != 768:
                        logger.warning(f"No valid embedding for chunk {i}")
                        failed_count += 1
                        continue

                    uuid_seed = f"{metadata.get('category','')}|{metadata.get('section','')}|{i}"
                    uid = generate_uuid5(uuid_seed)

                    try:
                        batch.add_object(
                            properties=metadata,
                            vector=embedding,
                            uuid=uid
                        )
                        successful_count += 1
                    except Exception as e:
                        logger.error(f"Failed to add object {i}: {e}")
                        failed_count += 1

            # NOW batch.failed_objects EXISTS!
            if hasattr(batch, 'failed_objects') and batch.failed_objects:
                for err in batch.failed_objects:
                    logger.error(f"Weaviate batch error: {err}")
                failed_count += len(batch.failed_objects)
                successful_count -= len(batch.failed_objects)

            successful_stores = successful_count
            failed_stores += failed_count
            logger.info(f"Successfully upserted {successful_stores} objects (failed: {failed_stores})")

        else:
            logger.warning(f"No text chunks to process for {filename}")
            failed_stores += total_objects

        # Final stats
        success_rate = (
            (successful_stores / (successful_stores + failed_stores) * 100)
            if (successful_stores + failed_stores) > 0 else 0
        )

        return {
            "total_objects": total_objects,
            "successful_stores": successful_stores,
            "failed_stores": failed_stores,
            "total_chunks": total_chunks,
            "images": images,
            "image_count": len(images),
            "success_rate": round(success_rate, 2)
        }

    def process_jsonl_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            json_objects = self.parse_jsonl_file(file_content)
            results = self.process_jsonl_batch(json_objects, filename)
            results.update({
                "filename": filename,
                "file_size_bytes": len(file_content),
                "processing_time": round(time.time() - start_time, 3)
            })
            logger.info(f"Processed {filename}: {results['successful_stores']}/{results['total_objects']} objects")
            return results
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in process_jsonl_file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process JSONL file: {str(e)}")


def get_jsonl_processor() -> JSONLProcessor:
    return JSONLProcessor()