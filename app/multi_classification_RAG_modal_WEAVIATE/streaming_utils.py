# app/multi_classification_RAG_modal/streaming_utils.py
# Updated messages to prioritize Hindi

import json
import asyncio
from typing import AsyncGenerator
from fastapi import WebSocket
from app.multi_classification_RAG_modal_WEAVIATE.rag_weaviate import get_vector_store, query_vector_store
from app.llm import get_google_response_stream

class StreamingChatbot:
    @staticmethod
    async def stream_rag_response(query: str, websocket: WebSocket):
        """Stream RAG response with real-time updates, prioritizing Hindi"""
        try:
            store = get_vector_store()
            top_k = 5
            results = query_vector_store(query, top_k=top_k)

            context_texts = []
            for match in results.get('matches', []):
                if 'text' in match['metadata']:
                    source = match['metadata'].get('source', 'text')
                    text = match['metadata']['text']
                    context_texts.append(text)

            if not context_texts:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "अपलोडेड JSONL में कोई प्रासंगिक जानकारी नहीं मिली।"  # Hindi priority
                }))
                return

            llm_response = ""
            async for chunk in get_google_response_stream(f"""
                Context: {"\n\n".join(context_texts)}
                Question: {query}
                हिंदी में उत्तर दें यदि संभव हो:
            """):
                llm_response += chunk
                await websocket.send_text(json.dumps({
                    "type": "stream",
                    "chunk": chunk,
                    "is_final": False
                }))
                await asyncio.sleep(0.01)
            await websocket.send_text(json.dumps({
                "type": "complete",
                "data": {"message": llm_response, "supportMessage": {"label": "और?", "options": ["Text Content", "Image Content"]}}
            }))
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"आरएजी प्रतिक्रिया में त्रुटि: {str(e)}"  # Hindi priority
            }))
            raise

    @staticmethod
    async def stream_text_response(text: str, websocket: WebSocket, chunk_size: int = 50):
        """Stream text response character by character for typing effect"""
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            await websocket.send_text(json.dumps({
                "type": "stream",
                "chunk": chunk,
                "is_final": i + chunk_size >= len(text)
            }))
            await asyncio.sleep(0.05)