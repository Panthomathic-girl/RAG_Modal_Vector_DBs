from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import json
from app.multimodal_chatbot.utils import _enforce_size_limit, _save_to_temp, _extract_pdf_text, _extract_pdf_images
from app.multimodal_chatbot.rag import get_vector_store, upsert_texts, query_vector_store
from app.llm import get_google_response_stream, get_image_description
import os

router = APIRouter(prefix="/multimodal_chat", tags=["Multimodal Chatbot"])

@router.get("/stream", tags=["Streaming"])
async def multimodal_stream(
    query: str = Query(..., description="Search query for text and image content in PDFs"),
    filename: str = Query(None, description="Optional filename to filter results (e.g., Classified_Work_Flow.pdf)"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to retrieve from vector store")
):
    """
    Stream responses for multimodal queries (text and images) using Server-Sent Events.
    """
    async def event_stream():
        try:
            filter = {"filename": {"$eq": filename}} if filename else None
            results = query_vector_store(query, top_k=top_k, filter=filter)
            
            context_texts = []
            for match in results.get('matches', []):
                if 'text' in match['metadata']:
                    source = match['metadata'].get('source', 'text')
                    text = match['metadata']['text']
                    if source == 'image':
                        text = f"Image on page {match['metadata'].get('page_number', 'unknown')}: {text}"
                    context_texts.append(text)
            
            support_message = {
                "label": "Would you like to know more about?",
                "options": [
                    "Text Content",
                    "Image Content",
                    "Upload Another PDF",
                    "Search Other Documents"
                ]
            }
            
            if context_texts:
                context = "\n\n".join(context_texts)
                rag_prompt = f"""Based on the following context from uploaded PDF documents (including text and image descriptions), answer the question. If the answer cannot be found, say so.

Context:
{context}

Question: {query}

Provide a concise answer based on the context:"""
                
                try:
                    full_response = ""
                    for chunk in get_google_response_stream(rag_prompt):
                        if chunk:
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    
                    final_data = {
                        "message": full_response,
                        "supportMessage": support_message
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                except Exception as llm_error:
                    error_data = {"error": f"Failed to generate LLM response: {str(llm_error)}"}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
            else:
                error_data = {"error": "No relevant information found in the uploaded PDFs."}
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
        
        except Exception as e:
            error_data = {"error": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/upload", tags=["Upload"])
async def upload_pdf(
    file: UploadFile = File(..., description="Upload a .pdf file"),
):
    """
    Upload a PDF, extract text and images, convert images to text via VLM, store in vector DB.
    Returns: JSON with text, images, and storage status.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        content = await _enforce_size_limit(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    name_lower = file.filename.lower()
    if not name_lower.endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only .pdf files are supported.")

    temp_path = None
    try:
        temp_path = _save_to_temp(content, suffix=".pdf")

        # Extract text
        text, page_count = _extract_pdf_text(temp_path)
        text = text or ""
        char_count = len(text)
        
        # Extract images
        images = _extract_pdf_images(content)
        image_descriptions = []
        for img_bytes, page_number in images:
            description = get_image_description(img_bytes)
            if description:
                image_descriptions.append({
                    "page_number": page_number,
                    "description": description,
                    "size_bytes": len(img_bytes)
                })

        # Store text in vector database
        text_vector_stored = False
        if text:
            print(f"Attempting to store text in vector database...")
            text_vector_stored = upsert_texts(text, file.filename, "pdf", source="text")
        
        # Store image descriptions in vector database
        image_vectors_stored = []
        for img_desc in image_descriptions:
            if img_desc["description"]:
                print(f"Attempting to store image description from page {img_desc['page_number']} in vector database...")
                metadata = {"page_number": img_desc["page_number"]}
                stored = upsert_texts(img_desc["description"], file.filename, "pdf", source="image", extra_metadata=metadata)
                image_vectors_stored.append(stored)

        # JSON response
        payload = {
            "filename": file.filename,
            "file_type": "pdf",
            "text": text,
            "text_size_bytes": len(text.encode('utf-8')),
            "text_chars": char_count,
            "text_vector_stored": text_vector_stored,
            "pages": page_count,
            "images": [
                {
                    "page_number": img_desc["page_number"],
                    "description": img_desc["description"],
                    "description_size_bytes": len(img_desc["description"].encode('utf-8')),
                    "image_size_bytes": img_desc["size_bytes"],
                    "vector_stored": image_vectors_stored[i] if i < len(image_vectors_stored) else False
                } for i, img_desc in enumerate(image_descriptions)
            ],
            "image_count": len(image_descriptions)
        }
        return JSONResponse(payload)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.get("/query", tags=["Query"])
async def query_documents(
    query: str = Query(..., description="Search query for text and image content in PDFs"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to retrieve from vector store"),
    include_llm_response: bool = Query(True, description="Include LLM-generated response")
):
    """
    Query the vector database for text and image content, with optional LLM response.
    """
    try:
        results = query_vector_store(query, top_k=top_k)
        
        formatted_results = []
        context_texts = []
        for match in results.get('matches', []):
            formatted_results.append({
                "id": match['id'],
                "score": match['score'],
                "metadata": match['metadata']
            })
            if 'text' in match['metadata']:
                source = match['metadata'].get('source', 'text')
                text = match['metadata']['text']
                if source == 'image':
                    text = f"Image on page {match['metadata'].get('page_number', 'unknown')}: {text}"
                context_texts.append(text)
        
        response_data = {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
        
        if include_llm_response and context_texts:
            try:
                context = "\n\n".join(context_texts)
                rag_prompt = f"""Based on the following context from uploaded PDF documents (including text and image descriptions), answer the question. If the answer cannot be found, say so.

Context:
{context}

Question: {query}

Provide a comprehensive answer based on the context:"""
                llm_response = get_google_response_stream(rag_prompt)
                response_data["llm_response"] = "".join([chunk for chunk in llm_response if chunk])
                response_data["context_used"] = len(context_texts)
            except Exception as llm_error:
                response_data["llm_error"] = f"Failed to generate LLM response: {str(llm_error)}"
                response_data["llm_response"] = None
        else:
            response_data["llm_response"] = None
            if not context_texts:
                response_data["llm_note"] = "No relevant information found in the uploaded PDFs."
        
        return JSONResponse(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/debug_vectors", tags=["Debug"])
async def debug_vectors():
    """
    Debug endpoint to check Pinecone index status.
    """
    try:
        store = get_vector_store()
        stats = store.describe_index_stats()
        # Convert IndexDescription to a JSON-serializable dictionary
        stats_dict = {
            "dimension": stats.get("dimension", 0),
            "index_fullness": stats.get("index_fullness", 0.0),
            "total_vector_count": stats.get("total_vector_count", 0),
            "namespaces": {
                ns: {
                    "vector_count": data.get("vector_count", 0)
                } for ns, data in stats.get("namespaces", {}).items()
            }
        }
        return JSONResponse({"stats": stats_dict})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)