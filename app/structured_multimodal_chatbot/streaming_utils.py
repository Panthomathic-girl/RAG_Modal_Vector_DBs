import json
import asyncio
from typing import AsyncGenerator
from fastapi import WebSocket
from app.structured_multimodal_chatbot.rag import get_vector_store, query_vector_store
from app.llm import get_google_response_stream

class StreamingChatbot:
    @staticmethod
    async def stream_rag_response(query: str, websocket: WebSocket):
        """Stream RAG response with real-time updates"""
        try:
            store = get_vector_store()
            top_k = 5
            results = query_vector_store(query, top_k=top_k)

            context_texts = []
            for match in results.get('matches', []):
                if 'text' in match['metadata']:
                    source = match['metadata'].get('source', 'text')
                    text = match['metadata']['text']
                    if "diagram" in query.lower() and source == 'image':
                        text = f"Image Description: {text}"
                    context_texts.append(text)

            if not context_texts:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "No relevant information found in the uploaded JSONL."
                }))
                return

            llm_response = ""
            async for chunk in get_google_response_stream(f"""
                Context: {"\n\n".join(context_texts)}
                Question: {query}
                Answer based on the context:
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
                "data": {"message": llm_response, "supportMessage": {"label": "More?", "options": ["Text Content", "Image Content"]}}
            }))
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Error in RAG response: {str(e)}"
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