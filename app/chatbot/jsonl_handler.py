"""
JSONL file handler for processing and storing data in vector database.

This module handles:
- JSONL file parsing and validation
- Data extraction and preprocessing
- Vector database storage with proper metadata
- Batch processing for large files
"""

import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Generator
from fastapi import HTTPException
from app.chatbot.rag import get_vector_store


class JSONLProcessor:
    """Handles JSONL file processing and vector database storage."""
    
    def __init__(self):
        self.vector_store = get_vector_store()
    
    def parse_jsonl_file(self, file_content: bytes) -> List[Dict[str, Any]]:
        """
        Parse JSONL file content and return list of JSON objects.
        
        Args:
            file_content: Raw bytes from uploaded file
            
        Returns:
            List of parsed JSON objects
            
        Raises:
            HTTPException: If file parsing fails
        """
        try:
            # Decode file content
            content = file_content.decode('utf-8')
            
            # Parse JSONL (each line is a separate JSON object)
            json_objects = []
            for line_num, line in enumerate(content.strip().split('\n'), 1):
                line = line.strip()
                if not line:  # Skip empty lines
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
                raise HTTPException(
                    status_code=400,
                    detail="No valid JSON objects found in file"
                )
                
            return json_objects
            
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be UTF-8 encoded"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse JSONL file: {str(e)}"
            )
    
    def extract_text_content(self, json_obj: Dict[str, Any]) -> str:
        """
        Extract text content from JSON object for embedding.
        
        Args:
            json_obj: Parsed JSON object
            
        Returns:
            Extracted text content
        """
        # Priority order for text extraction
        text_fields = ['text', 'content', 'description', 'body', 'summary', 'title']
        
        # Try to find text in common fields
        for field in text_fields:
            if field in json_obj and json_obj[field]:
                return str(json_obj[field]).strip()
        
        # If no specific text field, concatenate all string values
        text_parts = []
        for key, value in json_obj.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                # Recursively extract text from nested objects
                nested_text = self._extract_from_dict(value)
                if nested_text:
                    text_parts.append(f"{key}: {nested_text}")
        
        return " | ".join(text_parts) if text_parts else ""
    
    def _extract_from_dict(self, obj: Dict[str, Any]) -> str:
        """Recursively extract text from nested dictionary."""
        text_parts = []
        for key, value in obj.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                nested = self._extract_from_dict(value)
                if nested:
                    text_parts.append(nested)
        return " | ".join(text_parts)
    
    def create_metadata(self, json_obj: Dict[str, Any], filename: str, chunk_index: int = 0) -> Dict[str, Any]:
        """
        Create metadata for vector database storage.
        
        Args:
            json_obj: Original JSON object
            filename: Source filename
            chunk_index: Index of chunk (for large texts)
            
        Returns:
            Metadata dictionary
        """
        # Extract key fields for metadata - ensure no null values
        metadata = {
            "source_type": "jsonl",
            "filename": filename or "unknown",
            "chunk_index": chunk_index,
            "created_at": int(time.time()),
            "id": json_obj.get("id") or "",
            "url": json_obj.get("url") or "",
            "title": json_obj.get("title") or "",
            "section": json_obj.get("section") or "",
        }
        
        # Add any additional metadata from the JSON object, filtering out null values
        if "metadata" in json_obj and isinstance(json_obj["metadata"], dict):
            for key, value in json_obj["metadata"].items():
                # Only add non-null values
                if value is not None:
                    # Convert to string if it's not a basic type
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)
        
        # Add site information if available
        if json_obj.get("url"):
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(json_obj["url"])
                if parsed_url.netloc:
                    metadata["site"] = parsed_url.netloc
            except:
                pass
        
        # Clean up any remaining null values
        cleaned_metadata = {}
        for key, value in metadata.items():
            if value is not None:
                cleaned_metadata[key] = value
        
        return cleaned_metadata
    
    def chunk_large_text(self, text: str, max_chunk_size: int = 40000) -> List[str]:
        """
        Split large text into smaller chunks for embedding.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum size per chunk in bytes
            
        Returns:
            List of text chunks
        """
        if len(text.encode('utf-8')) <= max_chunk_size:
            return [text]
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len((current_chunk + paragraph).encode('utf-8')) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Single paragraph is too large, split by sentences
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if len((current_chunk + sentence).encode('utf-8')) > max_chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = sentence
                            else:
                                # Force split large sentence
                                chunks.append(sentence[:max_chunk_size])
                        else:
                            current_chunk += ". " + sentence if current_chunk else sentence
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_jsonl_batch(
        self, 
        json_objects: List[Dict[str, Any]], 
        filename: str,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Process JSONL objects in batches and store in vector database.
        
        Args:
            json_objects: List of parsed JSON objects
            filename: Source filename
            batch_size: Number of objects to process per batch
            
        Returns:
            Processing statistics
        """
        total_objects = len(json_objects)
        successful_stores = 0
        failed_stores = 0
        total_chunks = 0
        
        print(f"Processing {total_objects} JSON objects from {filename}")
        
        # Process in batches
        for batch_start in range(0, total_objects, batch_size):
            batch_end = min(batch_start + batch_size, total_objects)
            batch = json_objects[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}: objects {batch_start+1}-{batch_end}")
            
            # Prepare batch data
            batch_ids = []
            batch_texts = []
            batch_metadatas = []
            
            for obj_index, json_obj in enumerate(batch):
                try:
                    # Extract text content
                    text_content = self.extract_text_content(json_obj)
                    
                    if not text_content.strip():
                        print(f"Warning: No text content found in object {batch_start + obj_index + 1}")
                        failed_stores += 1
                        continue
                    
                    # Chunk large texts
                    text_chunks = self.chunk_large_text(text_content)
                    total_chunks += len(text_chunks)
                    
                    # Create vectors for each chunk
                    for chunk_index, chunk_text in enumerate(text_chunks):
                        # Generate unique ID
                        obj_id = json_obj.get("id", f"obj_{batch_start + obj_index}")
                        chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                        vector_id = f"{obj_id}_chunk_{chunk_index}_{chunk_hash}"
                        
                        # Create metadata
                        metadata = self.create_metadata(json_obj, filename, chunk_index)
                        metadata["text"] = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text
                        metadata["total_chunks"] = len(text_chunks)
                        
                        # Ensure all metadata values are valid for Pinecone
                        for key, value in list(metadata.items()):
                            if value is None:
                                del metadata[key]
                            elif not isinstance(value, (str, int, float, bool)):
                                metadata[key] = str(value)
                        
                        batch_ids.append(vector_id)
                        batch_texts.append(chunk_text)
                        batch_metadatas.append(metadata)
                        
                except Exception as e:
                    print(f"Error processing object {batch_start + obj_index + 1}: {e}")
                    failed_stores += 1
                    continue
            
            # Store batch in vector database
            if batch_ids:
                try:
                    self.vector_store.upsert_texts(
                        ids=batch_ids,
                        texts=batch_texts,
                        metadatas=batch_metadatas
                    )
                    successful_stores += len(batch_ids)
                    print(f"Successfully stored batch of {len(batch_ids)} vectors")
                    
                except Exception as e:
                    print(f"Error storing batch: {e}")
                    failed_stores += len(batch_ids)
        
        return {
            "total_objects": total_objects,
            "successful_stores": successful_stores,
            "failed_stores": failed_stores,
            "total_chunks": total_chunks,
            "success_rate": (successful_stores / (successful_stores + failed_stores) * 100) if (successful_stores + failed_stores) > 0 else 0
        }
    
    def process_jsonl_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Complete JSONL file processing pipeline.
        
        Args:
            file_content: Raw file content
            filename: Source filename
            
        Returns:
            Processing results and statistics
        """
        try:
            # Parse JSONL file
            json_objects = self.parse_jsonl_file(file_content)
            
            # Process and store in vector database
            results = self.process_jsonl_batch(json_objects, filename)
            
            # Add file-level statistics
            results.update({
                "filename": filename,
                "file_size_bytes": len(file_content),
                "processing_time": time.time()
            })
            
            return results
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process JSONL file: {str(e)}"
            )


def get_jsonl_processor() -> JSONLProcessor:
    """Get or create JSONL processor instance."""
    return JSONLProcessor()
