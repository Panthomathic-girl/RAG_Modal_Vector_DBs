import json
import hashlib
import time
from typing import List, Dict, Any
from fastapi import HTTPException
from app.structured_multimodal_chatbot.rag import get_vector_store, upsert_texts

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
        text_fields = ['text', 'content', 'description', 'body', 'summary', 'title']
        for field in text_fields:
            if field in json_obj and json_obj[field]:
                return str(json_obj[field]).strip()
        text_parts = []
        for key, value in json_obj.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(f"{key}: {value}")
        return " | ".join(text_parts) if text_parts else ""

    def create_metadata(self, json_obj: Dict[str, Any], filename: str, chunk_index: int, source: str = "text") -> Dict[str, Any]:
        return {
            "filename": filename,
            "source": source,
            "created_at": int(time.time()),
            "chunk_index": chunk_index,
            "total_chunks": 1
        }

    def process_jsonl_batch(self, json_objects: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
        total_objects = len(json_objects)
        successful_stores = 0
        failed_stores = 0
        total_chunks = 0
        images = []

        for obj_index, json_obj in enumerate(json_objects):
            try:
                text_content = self.extract_text_content(json_obj)
                if text_content:
                    chunks = [text_content]  # Single chunk for now
                    for chunk_index, chunk in enumerate(chunks):
                        vector_id = f"txt_{filename}_{obj_index}_{chunk_index}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                        metadata = self.create_metadata(json_obj, filename, chunk_index, "text")
                        metadata["text"] = chunk[:500] + "..." if len(chunk) > 500 else chunk
                        stored = upsert_texts(chunk, filename, "jsonl", "text", metadata)
                        if stored:
                            successful_stores += 1
                            total_chunks += 1
                        else:
                            failed_stores += 1

                if "images_base64" in json_obj and isinstance(json_obj["images_base64"], list):
                    for img_index, image_data in enumerate(json_obj["images_base64"]):
                        if "description" in image_data and image_data["description"]:
                            description = image_data["description"]
                            vector_id = f"img_{filename}_{obj_index}_{img_index}_{hashlib.md5(description.encode()).hexdigest()[:8]}"
                            metadata = self.create_metadata(json_obj, filename, img_index, "image")
                            metadata.update({
                                "text": description[:500] + "..." if len(description) > 500 else description,
                                "image_filename": image_data.get("filename", f"image_{img_index}.png")
                            })
                            stored = upsert_texts(description, filename, "jsonl", "image", metadata)
                            if stored:
                                successful_stores += 1
                                total_chunks += 1
                                images.append({"description": description, "size_bytes": 0})
                            else:
                                failed_stores += 1
            except Exception as e:
                print(f"Error processing object {obj_index + 1}: {e}")
                failed_stores += 1

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