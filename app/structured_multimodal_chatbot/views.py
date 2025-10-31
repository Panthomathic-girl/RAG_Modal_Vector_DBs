from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import json
from app.structured_multimodal_chatbot.utils import _enforce_size_limit
from app.structured_multimodal_chatbot.jsonl_handler import get_jsonl_processor
from app.structured_multimodal_chatbot.rag import get_vector_store, upsert_texts, query_vector_store
from app.llm import get_google_response_stream
import logging

router = APIRouter(prefix="/structured_multimodal_chat", tags=["Structured Multimodal Chatbot"])

@router.get("/stream")
async def structured_multimodal_stream(
    query: str = Query(..., description="Search query for text and image content in JSONL"),
    filename: str = Query(None, description="Optional filename to filter results (e.g., java_applet_pages.jsonl)"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to retrieve from vector store")
):
    """
    Stream responses for multimodal queries (text and images) from JSONL using Server-Sent Events.
    """
    async def event_stream():
        try:
            filter = {"filename": {"$eq": filename}} if filename else None
            results = query_vector_store(query, top_k=top_k, filter=filter)
            logging.debug(f"Query results for '{query}': {results}")

            context_texts = []
            for match in results.get('matches', []):
                if 'text' in match['metadata']:
                    source = match['metadata'].get('source', 'text')
                    text = match['metadata']['text']
                    if "diagram" in query.lower() and source == 'image':
                        text = f"Image Description: {text}"
                    context_texts.append(text)

            support_message = {
                "label": "Would you like to know more about?",
                "options": ["Text Content", "Image Content", "Upload Another JSONL", "Search Other Documents"]
            }

            if context_texts:
                context = "\n\n".join(context_texts)
                rag_prompt = f"""Based on the following context from uploaded JSONL documents (including text and image descriptions), answer the question. If the answer cannot be found, say so.

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
                    final_data = {"message": full_response, "supportMessage": support_message}
                    yield f"data: {json.dumps(final_data)}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                except Exception as llm_error:
                    error_data = {"error": f"Failed to generate LLM response: {str(llm_error)}"}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
            else:
                error_data = {"error": "No relevant information found in the uploaded JSONL."}
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            error_data = {"error": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/upload")
async def upload_jsonl(file: UploadFile = File(..., description="Upload a .jsonl file")):
    """
    Upload a JSONL file, extract text and base64 images, store descriptions in vector DB.
    Returns: JSON with text, images, and storage status.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        content = _enforce_size_limit(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    name_lower = file.filename.lower()
    if not name_lower.endswith(".jsonl"):
        raise HTTPException(status_code=415, detail="Only .jsonl files are supported.")

    try:
        processor = get_jsonl_processor()
        results = processor.process_jsonl_file(content, file.filename)
        logging.info(f"Processed {file.filename}: {results}")
        
        payload = {
            "filename": file.filename,
            "file_type": "jsonl",
            "total_objects": results["total_objects"],
            "successful_stores": results["successful_stores"],
            "failed_stores": results["failed_stores"],
            "total_chunks": results["total_chunks"],
            "image_count": results["image_count"],
            "images": results["images"],
            "success_rate": results["success_rate"]
        }
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process JSONL file: {str(e)}")

@router.get("/query")
async def query_documents(
    query: str = Query(..., description="Search query for text and image content in JSONL"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to retrieve from vector store"),
    include_llm_response: bool = Query(True, description="Include LLM-generated response")
):
    """
    Query the vector database for text and image content, with optional LLM response.
    """
    try:
        results = query_vector_store(query, top_k=top_k)
        logging.debug(f"Query results for '{query}': {results}")

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
                if "diagram" in query.lower() and source == 'image':
                    text = f"Image Description: {text}"
                context_texts.append(text)

        response_data = {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }

        if include_llm_response and context_texts:
            try:
                context = "\n\n".join(context_texts)
                rag_prompt = f"""Based on the following context from uploaded JSONL documents (including text and image descriptions), answer the question. If the answer cannot be found, say so.

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
                response_data["llm_note"] = "No relevant information found in the uploaded JSONL."

        return JSONResponse(response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")