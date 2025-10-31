# app/multi_classification_RAG_modal_WEAVIATE/views.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import json
from app.multi_classification_RAG_modal_WEAVIATE.utils import _enforce_size_limit
from app.multi_classification_RAG_modal_WEAVIATE.jsonl_handler import get_jsonl_processor
from app.multi_classification_RAG_modal_WEAVIATE.rag_weaviate import get_vector_store, upsert_texts, query_vector_store
from app.llm import get_google_response_stream, get_google_response
import logging

router = APIRouter(prefix="/multi_classification_rag_weaviate", tags=["Multi Classification RAG Weaviate Modal"])

def classify_query(query: str) -> str:
    """
    Classify the query into a category using LLM for multi-classification.
    Prioritizes Hindi-related categories.
    """
    categories = [
        "Login & Profile Related",
        "Ad Booking Process Related",
        "Rate Card / Pricing Related",
        "Creative / Ad Material Related",
        "Approval & Payment Related",
        "Reports & Analytics",
        "Technical / Error Handling",
        "Help & Training",
        "Notifications & Updates",
        "General",
        "Intro",
        "Site Navigation (labels)",
        "Legacy and Mission",
        "Reach and Editions",
        "Engaging the Non-News Audience",
        "Editorial Excellence",
        "Modern Coverage"
    ]
    prompt = f"""Classify the following query into one of these categories. If the query is in Hindi or related to FAQs, prioritize Hindi categories. Output only the category name.

Categories: {', '.join(categories)}

Query: {query}

"""
    category = get_google_response(prompt).strip()
    return category if category in categories else None

@router.get("/stream")
async def multi_classification_stream(
    query: str = Query(..., description="Search query for text content in merged JSONL (supports Hindi/English)"),
    filename: str = Query(None, description="Optional filename to filter results (e.g., merged_rag_structured.jsonl)"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to retrieve from vector store")
):
    """
    Stream responses for queries from merged JSONL using Server-Sent Events. Supports bilingual queries.
    """
    async def event_stream():
        try:
            category = classify_query(query)  # Classify first
            filter = {"filename": {"$eq": filename}} if filename else {}
            if category:
                filter["category"] = category  # Add classification filter

            results = query_vector_store(query, top_k=top_k, filter=filter)
            logging.debug(f"Query results for '{query}': {results}")

            context_texts = []
            for match in results.get('matches', []):
                if 'text' in match['metadata']:
                    source = match['metadata'].get('source', 'text')
                    text = match['metadata']['text']
                    context_texts.append(text)

            support_message = {
                "label": "क्या आप और जानना चाहेंगे?",  # Prioritize Hindi
                "options": ["Text Content", "Image Content", "Upload Another JSONL", "Search Other Documents"]
            }

            if context_texts:
                context = "\n\n".join(context_texts)
                rag_prompt = f"""नीचे दिए गए संदर्भ से (जो हिंदी और अंग्रेजी में है), प्रश्न का उत्तर दें। यदि उत्तर नहीं मिलता, तो कहें। हिंदी में उत्तर दें यदि संभव हो, अन्यथा अंग्रेजी में।

Context:
{context}

Question: {query}

संक्षिप्त उत्तर प्रदान करें:"""  # Updated prompt to prioritize Hindi
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
                error_data = {"error": "अपलोडेड JSONL में कोई प्रासंगिक जानकारी नहीं मिली।"}  # Hindi error
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            error_data = {"error": f"एक त्रुटि हुई: {str(e)}"}  # Hindi error
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/upload")
async def upload_jsonl(file: UploadFile = File(..., description="Upload a .jsonl file like merged_rag_structured.jsonl")):
    """
    Upload a JSONL file, extract text, store in vector DB. Supports bilingual content.
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
    query: str = Query(..., description="Search query for text content in JSONL (supports Hindi/English)"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to retrieve from vector store"),
    include_llm_response: bool = Query(True, description="Include LLM-generated response")
):
    """
    Query the vector database for text content, with optional LLM response. Bilingual support.
    """
    try:
        category = classify_query(query)  # Classify first
        filter = {}
        if category:
            filter["category"] = category

        results = query_vector_store(query, top_k=top_k, filter=filter)
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
                context_texts.append(text)

        response_data = {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }

        if include_llm_response and context_texts:
            try:
                context = "\n\n".join(context_texts)
                rag_prompt = f"""नीचे दिए गए संदर्भ से (जो हिंदी और अंग्रेजी में है), प्रश्न का उत्तर दें। यदि उत्तर नहीं मिलता, तो कहें। हिंदी में उत्तर दें यदि संभव हो, अन्यथा अंग्रेजी में।

Context:
{context}

Question: {query}

व्यापक उत्तर प्रदान करें:"""  # Updated prompt to prioritize Hindi
                llm_response = get_google_response_stream(rag_prompt)
                response_data["llm_response"] = "".join([chunk for chunk in llm_response if chunk])
                response_data["context_used"] = len(context_texts)
            except Exception as llm_error:
                response_data["llm_error"] = f"Failed to generate LLM response: {str(llm_error)}"
                response_data["llm_response"] = None
        else:
            response_data["llm_response"] = None
            if not context_texts:
                response_data["llm_note"] = "अपलोडेड JSONL में कोई प्रासंगिक जानकारी नहीं मिली।"  # Hindi note

        return JSONResponse(response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")