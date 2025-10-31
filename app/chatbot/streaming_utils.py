import json
import asyncio
from typing import AsyncGenerator
from fastapi import WebSocket
from app.chatbot.agents import get_intent, rag_agent, customer_agent, agency_agent, other_agent
from app.chatbot.rag import get_vector_store


class StreamingChatbot:
    @staticmethod
    async def stream_rag_response(query: str, websocket: WebSocket):
        """Stream RAG response with real-time updates"""
        try:
            store = get_vector_store()
            top_k = 5
            source_filter = None
            include_llm_response = True

            # Build filter if source_filter is provided
            filter_dict = None
            if source_filter:
                filter_dict = {"source_type": {"$eq": source_filter}}
            
            results = store.query_by_text(
                query, 
                top_k=top_k,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            context_texts = []
            
            for match in results.get('matches', []):
                result_item = {
                    "id": match['id'],
                    "score": match['score'],
                    "metadata": match['metadata']
                }
                formatted_results.append(result_item)
                
                # Collect context for LLM
                if 'text' in match['metadata']:
                    context_texts.append(match['metadata']['text'])
            
            # Generate LLM response if we have context
            if include_llm_response and context_texts:
                llm_response = await rag_agent(query, context_texts)
                return llm_response
            else:
                return "I couldn't find relevant information to answer your question."
                
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Error in RAG response: {str(e)}"
            }))
            raise Exception(f"RAG query failed: {str(e)}")

    @classmethod
    async def stream_ai_response(cls, query: str, intent: str, websocket: WebSocket):
        """Stream AI response based on intent"""
        
        if intent == "refund":
            message = {
                "label": "would you like to know more about?",
                "options": ["Rajasthan Patrika", "Refund Policies", "Ad Booking"]
            }
            response = "To process a refund, please provide your booking details or contact our support team at support@rajasthanpatrika.com. Refunds are typically processed within 5-7 business days after verification."
            return response, message

        elif intent == "rp":
            response = await cls.stream_rag_response(query, websocket)
            message = {
                "label": "would you like to know more about?",
                "options": ["Rajasthan Patrika", "Refund Policies", "Ad Booking"]
            }
            return response, message

        elif intent == "ad_booking":
            message = {
                "label": "For ad booking, please specify the category you are interested in:",
                "options": ["Customer", "Agency"]
            }
            response = "To help you with ad booking, could you please let me know which category you want to book an ad for? (Customer, Agency)"
            return response, message
            
        elif intent == "customer":
            message = {
                "label": "Would you like to know the ad booking flow for any other category like Agency, or want to know about Rajasthan Patrika or refund policy?",
                "options": ["Agency", "Rajasthan Patrika", "Refund Policy"]
            }
            response = await customer_agent(query)
            return response, message
            
        elif intent == "agency":
            message = {
                "label": "Would you like to know the ad booking flow for any other category like Customer, or want to know about Rajasthan Patrika or refund policy?",
                "options": ["Customer", "Rajasthan Patrika", "Refund Policy"]
            }
            response = await agency_agent(query)
            return response, message
            
        elif intent == "other":
            message = {
                "label": "Would you like to know the ad booking flow for any other category like Customer, Agency, or want to know about Rajasthan Patrika or refund policy?",
                "options": ["Customer", "Agency", "Rajasthan Patrika", "Refund Policy"]
            }
            response = await other_agent(query)
            return response, message
            
        else:
            message = {
                "label": "Would you like to know the ad booking flow for any other category like Customer, Agency, or want to know about Rajasthan Patrika or refund policy?",
                "options": ["Customer", "Agency", "Rajasthan Patrika", "Refund Policy"]
            }
            response = await other_agent(query)
            return response, message

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
            await asyncio.sleep(0.05)  # Small delay for typing effect
