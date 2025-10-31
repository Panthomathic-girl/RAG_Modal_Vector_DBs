from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import json
from app.unstructured_chatbot.utils import _enforce_size_limit, _save_to_temp, _extract_pdf_text
from app.unstructured_chatbot.rag import get_vector_store, upsert_texts
from app.llm import get_google_response
import os

router = APIRouter(prefix="/unstructured_chat", tags=["Unstructured Chatbot"])


@router.get("/unstructured_stream")
async def unstructured_stream(
    query: str = Query(..., description="Search query to retrieve and answer based on uploaded PDFs"),
    filename: str = Query(None, description="Optional filename to filter results (e.g., Classified_Work_Flow.pdf)"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to retrieve from vector store")
):
    """
    Stream chat responses using Server-Sent Events (SSE) based on uploaded PDFs.
    Provides real-time streaming of AI responses for better user experience.
    Optionally filters results by filename.
    """
    async def event_stream():
        try:
            # Query vector store with optional filename filter
            store = get_vector_store()
            filter = {"filename": {"$eq": filename}} if filename else None
            results = store.query_by_text(query, top_k=top_k, filter=filter)
            
            # Format results
            context_texts = []
            for match in results.get('matches', []):
                if 'text' in match['metadata']:
                    context_texts.append(match['metadata']['text'])
            
            # Define support message
            support_message = {
                "label": "Would you like to know more about?",
                "options": [
                    "Classified Work Flow",
                    "PDF Content Details",
                    "Upload Another PDF",
                    "Search Other Documents"
                ]
            }
            
            # Generate LLM response if context is available
            if context_texts:
                context = "\n\n".join(context_texts)
                rag_prompt = f"""Based on the following context from uploaded PDF documents, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Please provide a concise and accurate answer based on the context above:"""
                
                try:
                    # Stream the LLM response
                    full_response = ""
                    for chunk in get_google_response(rag_prompt, stream=True):  # Assuming get_google_response supports streaming
                        if chunk:
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    
                    # Send final complete response with support message
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
                error_data = {"error": "No relevant information found in the uploaded PDFs to answer your query."}
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
        
        except Exception as e:
            error_data = {"error": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/unstructured")
async def unstructured_query(
    query: str = Query(..., description="Search query to retrieve and answer based on uploaded PDFs"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to retrieve from vector store")
):
    """
    Query the vector database with a search string and get an LLM-generated response based on uploaded PDFs.
    Returns the query, LLM response, and metadata about retrieved documents.
    """
    try:
        store = get_vector_store()
        results = store.query_by_text(query, top_k=top_k)
        
        # Format results
        formatted_results = []
        context_texts = []
        
        for match in results.get('matches', []):
            formatted_results.append({
                "id": match['id'],
                "score": match['score'],
                "metadata": match['metadata']
            })
            # Collect context for LLM
            if 'text' in match['metadata']:
                context_texts.append(match['metadata']['text'])
        
        response_data = {
            "statusCode": 200,
            "message": "Query processed successfully",
            "data": {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
        }
        
        # Generate LLM response if context is available
        if context_texts:
            try:
                # Create context for LLM
                context = "\n\n".join(context_texts)
                
                # Create RAG prompt
                rag_prompt = f"""Based on the following context from uploaded PDF documents, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Please provide a concise and accurate answer based on the context above:"""
                
                # Get LLM response
                llm_response = get_google_response(rag_prompt)
                
                response_data["data"]["message"] = llm_response
                response_data["data"]["context_used"] = len(context_texts)
                
            except Exception as llm_error:
                response_data["data"]["message"] = None
                response_data["data"]["llm_error"] = f"Failed to generate LLM response: {str(llm_error)}"
        else:
            response_data["data"]["message"] = "No relevant information found in the uploaded PDFs to answer your query."
            response_data["data"]["note"] = "No context available for LLM response"
        
        return JSONResponse(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(..., description="Upload a .pdf file"),
):
    """
    Upload a PDF and extract text, store in vector DB.
    Returns JSON with filename, text, and stats.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # Read & size-check
    try:
        content = _enforce_size_limit(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    # Extension sniffing
    name_lower = file.filename.lower()
    if name_lower.endswith(".pdf"):
        suffix = ".pdf"
        ftype = "pdf"
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type. Only .pdf is allowed.")

    # Persist to temp for libraries that want a path
    temp_path = None
    try:
        temp_path = _save_to_temp(content, suffix=suffix)

        text, page_count = _extract_pdf_text(temp_path)

        text = text or ""  # ensure string
        char_count = len(text)
        
        # Store in vector database
        print(f"Attempting to store text in vector database...")
        vector_stored = upsert_texts(text, file.filename, ftype)

        # JSON response with all stats included
        payload = {
            "filename": file.filename,
            "file_type": ftype,
            "text": text,
            "size_bytes": len(content),
            "pages": page_count,
            "chars": char_count,
            "vector_stored": vector_stored,
            "text_size_bytes": len(text.encode('utf-8')),
        }
        return JSONResponse(payload)

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/query")
async def query_documents(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    include_llm_response: bool = Query(True, description="Include LLM-generated response")
):
    """
    Query the vector database for similar documents and optionally generate an LLM response.
    """
    try:
        store = get_vector_store()
        results = store.query_by_text(query, top_k=top_k)
        
        # Format results
        formatted_results = []
        context_texts = []
        
        for match in results.get('matches', []):
            formatted_results.append({
                "id": match['id'],
                "score": match['score'],
                "metadata": match['metadata']
            })
            # Collect context for LLM
            if 'text' in match['metadata']:
                context_texts.append(match['metadata']['text'])
        
        response_data = {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
        
        # Generate LLM response if requested and we have context
        if include_llm_response and context_texts:
            try:
                # Create context for LLM
                context = "\n\n".join(context_texts)
                
                # Create RAG prompt
                rag_prompt = f"""Based on the following context from uploaded documents, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above:"""
                
                # Get LLM response
                llm_response = get_google_response(rag_prompt)
                
                response_data["llm_response"] = llm_response
                response_data["context_used"] = len(context_texts)
                
            except Exception as llm_error:
                response_data["llm_error"] = f"Failed to generate LLM response: {str(llm_error)}"
                response_data["llm_response"] = None
        else:
            response_data["llm_response"] = None
            if not context_texts:
                response_data["llm_note"] = "No context available for LLM response"
        
        return JSONResponse(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")