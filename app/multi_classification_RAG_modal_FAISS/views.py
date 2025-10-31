# app/multi_classification_RAG_modal/views.py
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
import logging
from typing import Dict, Any

from app.multi_classification_RAG_modal_FAISS.utils import _enforce_size_limit
from app.multi_classification_RAG_modal_FAISS.rag_faiss import FAISSRetriever

router = APIRouter(tags=["Multi Classification RAG Modal FAISS"])


# ----------------------------------------------------------------------
# 1. Upload → (re)build FAISS index
# ----------------------------------------------------------------------
@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")
    content = _enforce_size_limit(file)
    if not file.filename.lower().endswith(".jsonl"):
        raise HTTPException(415, "Only .jsonl")

    # Use tempfile for cross-platform temp file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        # Re-build index
        FAISSRetriever.build_index(tmp_path, force_rebuild=True)
        logging.info(f"Uploaded & indexed {file.filename}")
    finally:
        # Always clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logging.warning(f"Failed to delete temp file {tmp_path}: {e}")

    return JSONResponse({
        "filename": file.filename,
        "status": "indexed",
        "message": "FAISS index rebuilt successfully"
    })


# ----------------------------------------------------------------------
# 2. /stream – multilingual retrieval + typing effect
# ----------------------------------------------------------------------
@router.get("/stream")
async def stream(
    query: str = Query(..., description="Hindi / Hinglish / English query"),
    top_k: int = Query(5, ge=1, le=20)
):
    async def gen():
        try:
            support = {
                "label": "क्या आप और जानना चाहेंगे?",
                "options": ["Text Content", "Image Content", "Upload Another JSONL", "Search Other Documents"]
            }

            results = FAISSRetriever.retrieve(query, top_k=top_k)

            if not results:
                yield f"data: {json.dumps({'error': 'कोई प्रासंगिक जानकारी नहीं मिली।'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return

            # Use the **best** chunk as the answer
            best = results[0]
            answer = best["metadata"]["text"]
            score = best["score"]

            logging.info(
                "RAG HIT | query:%s | id:%s | score:%.3f | title:%s",
                query, best["metadata"]["id"], score, best["metadata"].get("title","")
            )

            # Typing effect
            chunk_sz = 45
            for i in range(0, len(answer), chunk_sz):
                yield f"data: {json.dumps({'chunk': answer[i:i+chunk_sz]})}\n\n"
                await asyncio.sleep(0.05)

            yield f"data: {json.dumps({'message': answer, 'supportMessage': support})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        except Exception as e:
            err = f"त्रुटि: {str(e)}"
            logging.error(err)
            yield f"data: {json.dumps({'error': err})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ----------------------------------------------------------------------
# 3. /query – JSON debug endpoint
# ----------------------------------------------------------------------
@router.get("/query")
async def query(
    query: str = Query(...),
    top_k: int = Query(5, ge=1, le=20)
):
    results = FAISSRetriever.retrieve(query, top_k=top_k)
    resp = {
        "query": query,
        "results": [
            {"score": r["score"], "metadata": r["metadata"]} for r in results
        ],
        "total": len(results)
    }
    if results:
        resp["answer"] = results[0]["metadata"]["text"]
    else:
        resp["note"] = "कोई जानकारी नहीं मिली।"
    return JSONResponse(resp)