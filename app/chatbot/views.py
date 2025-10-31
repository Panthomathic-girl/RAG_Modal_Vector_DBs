from fastapi import FastAPI, File, UploadFile, HTTPException, Query, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from app.chatbot.jsonl_handler import get_jsonl_processor
from app.chatbot.streaming_utils import StreamingChatbot
from app.chatbot.agents import get_intent, rag_agent_stream, customer_agent_stream, agency_agent_stream, other_agent_stream
from app.llm import get_google_response, get_google_response_stream
from app.chatbot.utils import Chatbot
from app.chatbot.rag import get_vector_store

router = APIRouter(prefix="/chat", tags=["Tutorial"])


# WebSocket endpoint for real-time chat streaming
@router.websocket("/ws2")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            query = message_data.get("query", "")
            
            if not query:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "No query provided"
                }))
                continue
            
            
            intent = "rp"
            # Get streaming AI response
            response, support_message = await StreamingChatbot.stream_ai_response(query, intent, websocket)
            
            # Stream the response text with typing effect
            await StreamingChatbot.stream_text_response(response, websocket)
            
            # Send the complete response with support message
            await websocket.send_text(json.dumps({
                "type": "complete",
                "data": {
                    "message": response,
                    "supportMessage": support_message
                }
            }))
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"An error occurred: {str(e)}"
        }))
        print(f"WebSocket error: {e}")


# WebSocket endpoint for real-time chat streaming
@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            query = message_data.get("query", "")
            
            if not query:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "No query provided"
                }))
                continue
            
            
            
            # Handle special cases with immediate responses
            if query.lower() == "rajasthan patrika":
                response_data = {
                    "message": "Rajasthan Patrika has a readership of over 2.59 crore, positioning it 26th among the top 50 paid newspapers worldwide (WAN-IFRA). It is the 5th largest newspaper in India, according to AMIC UNESCO. Rajasthan Patrika is the only Hindi daily with a significant presence in southern India.\n\nIt is part of the Patrika Group, which reaches 50% of India's Hindi-speaking population and has extended its reach across 8 states with 38 editions. In MPCG, the publication is released as \"Patrika\".\n\nRajasthan Patrika is known as the \"Newspaper with a Soul,\" committed to comprehensive coverage of national events, local stories, and daily issues, addressing every facet of society and community with unwavering dedication.\n\nNotable associations and events include:\n*   Amitabh Bachchan and Prakash Jha in a Satyagraha Patrika TVC.\n*   Deepika Padukone TVC.\n*   Ranveer Singh (Catchnews TVC).\n*   Priyanka Chopra.\n*   Dr. APJ Abdul Kalam's visit to the Rajasthan Patrika office.\n*   Parineeti Chopra's visit to Rajasthan Patrika.\n*   Mac Rajasthan Patrika.\n\nRajasthan Patrika initiates sharp and effective reactions through regular series of campaigns, taking up problems not noticed by the administration or raised by others, leading to responsive administrative action.\n\nPatrika.com, part of Patrika Digital, is one of India's fastest-growing online news sites, building a loyal audience on the web and mobile. Patrika Digital boasts over 135 million page views and engages more than 39.7 million unique visitors from across the world on various platforms, creating cutting-edge advertising solutions for client visibility.",
                    "supportMessage": {
                        "label": "would you like to know more about?",
                        "options": ["Rajasthan Patrika", "Refund Policies", "Ad Booking"]
                    }
                }
                await websocket.send_text(json.dumps({
                    "type": "complete",
                    "data": response_data
                }))
                continue
                
            elif query.lower() == "refund policies":
                response_data = {
                    "message": "To process a refund, please provide your booking details or contact our support team at support@rajasthanpatrika.com. Refunds are typically processed within 5-7 business days after verification.",
                    "supportMessage": {
                        "label": "would you like to know more about?",
                        "options": ["Rajasthan Patrika", "Refund Policies", "Ad Booking"]
                    }
                }
                await websocket.send_text(json.dumps({
                    "type": "complete",
                    "data": response_data
                }))
                continue
                
            elif query.lower() == "ad booking":
                response_data = {
                    "message": "To help you with ad booking, could you please let me know which category you want to book an ad for? (Customer, Agency)",
                    "supportMessage": {
                        "label": "For ad booking, please specify the category you are interested in:",
                        "options": ["Customer", "Agency"]
                    }
                }
                await websocket.send_text(json.dumps({
                    "type": "complete",
                    "data": response_data
                }))
                continue
            
            # For AI responses, use streaming chatbot
            # Get intent (for internal processing only)
            # intent = await get_intent(query)
            intent = await Chatbot.intent_classification(query)
            print("Intent: ", intent)
            
            
            # Get streaming AI response
            response, support_message = await StreamingChatbot.stream_ai_response(query, intent, websocket)
            
            # Stream the response text with typing effect
            await StreamingChatbot.stream_text_response(response, websocket)
            
            # Send the complete response with support message
            await websocket.send_text(json.dumps({
                "type": "complete",
                "data": {
                    "message": response,
                    "supportMessage": support_message
                }
            }))
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"An error occurred: {str(e)}"
        }))
        print(f"WebSocket error: {e}")




@router.post("/")
async def chat(query: str = Query(..., description="Search query")):

    if query.lower() == "rajasthan patrika":
        data = {
            "message": "Rajasthan Patrika has a readership of over 2.59 crore, positioning it 26th among the top 50 paid newspapers worldwide (WAN-IFRA). It is the 5th largest newspaper in India, according to AMIC UNESCO. Rajasthan Patrika is the only Hindi daily with a significant presence in southern India.\n\nIt is part of the Patrika Group, which reaches 50% of India's Hindi-speaking population and has extended its reach across 8 states with 38 editions. In MPCG, the publication is released as “Patrika”.\n\nRajasthan Patrika is known as the \"Newspaper with a Soul,\" committed to comprehensive coverage of national events, local stories, and daily issues, addressing every facet of society and community with unwavering dedication.\n\nNotable associations and events include:\n*   Amitabh Bachchan and Prakash Jha in a Satyagraha Patrika TVC.\n*   Deepika Padukone TVC.\n*   Ranveer Singh (Catchnews TVC).\n*   Priyanka Chopra.\n*   Dr. APJ Abdul Kalam's visit to the Rajasthan Patrika office.\n*   Parineeti Chopra's visit to Rajasthan Patrika.\n*   Mac Rajasthan Patrika.\n\nRajasthan Patrika initiates sharp and effective reactions through regular series of campaigns, taking up problems not noticed by the administration or raised by others, leading to responsive administrative action.\n\nPatrika.com, part of Patrika Digital, is one of India's fastest-growing online news sites, building a loyal audience on the web and mobile. Patrika Digital boasts over 135 million page views and engages more than 39.7 million unique visitors from across the world on various platforms, creating cutting-edge advertising solutions for client visibility.",
            "supportMessage": {
                "label": "would you like to know more about?",
                "options": [
                    "Rajasthan Patrika",
                    "Refund Policies",
                    "Ad Booking"
                ]
            }
        }

        final_response = {
            "statusCode": 200,
            "message": "LLM response generated successfully",
            "data": data
        }
        return final_response

    elif query.lower() == "refund policies":
        data = {
            "message": "To process a refund, please provide your booking details or contact our support team at support@rajasthanpatrika.com. Refunds are typically processed within 5-7 business days after verification.",
            "supportMessage": {
                "label": "would you like to know more about?",
                "options": ["Rajasthan Patrika", "Refund Policies", "Ad Booking"]
            }
        }

        final_response = {
            "statusCode": 200,
            "message": "LLM response generated successfully",
            "data": data
        }
        return final_response

    elif query.lower() == "ad booking":
        data = {
            "message": "To help you with ad booking, could you please let me know which category you want to book an ad for? (Customer, Agency)",
            "supportMessage": {
                "label": "For ad booking, please specify the category you are interested in:",
                "options": ["Customer", "Agency"]
            }
        }

        final_response = {
            "statusCode": 200,
            "message": "LLM response generated successfully",
            "data": data
        }
        return final_response
        

    intent = await get_intent(query)
    print("Intent: ", intent)
    response, message = await Chatbot.ai_response(query, intent)
    data = {
        "message": response,
        "supportMessage": message
    }
    final_response = {
        "statusCode": 200,
        "message": "LLM response generated successfully",
        "data": data
    }
    return final_response



@router.get("/stream2")
async def chat_stream(query: str = Query(..., description="Search query")):
    """
    Stream chat responses using Server-Sent Events (SSE).
    Provides real-time streaming of AI responses for better user experience.
    """
    
    async def event_stream():
        try:
            
            
            # Determine support message based on intent
            support_message = {}
            stream_generator = None
            
            intent = "rp"
            
                
            if intent == "rp":
                support_message = {
                    "label": "would you like to know more about?",
                    "options": ["Rajasthan Patrika", "Refund Policies", "Ad Booking"]
                }
                # Get RAG context
                try:
                    store = get_vector_store()
                    results = store.query_by_text(query, top_k=5)
                    context_texts = []
                    for match in results.get('matches', []):
                        if 'text' in match['metadata']:
                            context_texts.append(match['metadata']['text'])
                    
                    if context_texts:
                        stream_generator = rag_agent_stream(query, context_texts)
                    else:
                        stream_generator = other_agent_stream(query)
                except Exception as e:
                    print(f"RAG error: {e}")
                    stream_generator = other_agent_stream(query)
                    
            
                
            else:  # "other" intent
                support_message = {
                    "label": "Would you like to know the ad booking flow for any other category like Customer, Agency, or want to know about Rajasthan Patrika or refund policy?",
                    "options": ["Customer", "Agency", "Rajasthan Patrika", "Refund Policy"]
                }
                stream_generator = other_agent_stream(query)
            
            # Stream the response if we have a generator
            if stream_generator:
                full_response = ""
                for chunk in stream_generator:
                    if chunk:
                        full_response += chunk
                        # Send each chunk as SSE
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Send final complete response with support message
                final_data = {
                    "message": full_response,
                    "supportMessage": support_message
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
        except Exception as e:
            error_data = {"error": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

    


@router.get("/stream")
async def chat_stream(query: str = Query(..., description="Search query")):
    """
    Stream chat responses using Server-Sent Events (SSE).
    Provides real-time streaming of AI responses for better user experience.
    """
    
    async def event_stream():
        try:
            # Handle special predefined responses (send complete without streaming)
            if query.lower() == "rajasthan patrika":
                response_data = {
                    "message": "Rajasthan Patrika has a readership of over 2.59 crore, positioning it 26th among the top 50 paid newspapers worldwide (WAN-IFRA). It is the 5th largest newspaper in India, according to AMIC UNESCO. Rajasthan Patrika is the only Hindi daily with a significant presence in southern India.\n\nIt is part of the Patrika Group, which reaches 50% of India's Hindi-speaking population and has extended its reach across 8 states with 38 editions. In MPCG, the publication is released as \"Patrika\".\n\nRajasthan Patrika is known as the \"Newspaper with a Soul,\" committed to comprehensive coverage of national events, local stories, and daily issues, addressing every facet of society and community with unwavering dedication.\n\nNotable associations and events include:\n*   Amitabh Bachchan and Prakash Jha in a Satyagraha Patrika TVC.\n*   Deepika Padukone TVC.\n*   Ranveer Singh (Catchnews TVC).\n*   Priyanka Chopra.\n*   Dr. APJ Abdul Kalam's visit to the Rajasthan Patrika office.\n*   Parineeti Chopra's visit to Rajasthan Patrika.\n*   Mac Rajasthan Patrika.\n\nRajasthan Patrika initiates sharp and effective reactions through regular series of campaigns, taking up problems not noticed by the administration or raised by others, leading to responsive administrative action.\n\nPatrika.com, part of Patrika Digital, is one of India's fastest-growing online news sites, building a loyal audience on the web and mobile. Patrika Digital boasts over 135 million page views and engages more than 39.7 million unique visitors from across the world on various platforms, creating cutting-edge advertising solutions for client visibility.",
                    "supportMessage": {
                        "label": "would you like to know more about?",
                        "options": ["Rajasthan Patrika", "Refund Policies", "Ad Booking"]
                    }
                }
                yield f"data: {json.dumps(response_data)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return
                
            elif query.lower() == "refund policies":
                response_data = {
                    "message": "To process a refund, please provide your booking details or contact our support team at support@rajasthanpatrika.com. Refunds are typically processed within 5-7 business days after verification.",
                    "supportMessage": {
                        "label": "would you like to know more about?",
                        "options": ["Rajasthan Patrika", "Refund Policies", "Ad Booking"]
                    }
                }
                yield f"data: {json.dumps(response_data)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return
                
            elif query.lower() == "ad booking":
                response_data = {
                    "message": "To help you with ad booking, could you please let me know which category you want to book an ad for? (Customer, Agency)",
                    "supportMessage": {
                        "label": "For ad booking, please specify the category you are interested in:",
                        "options": ["Customer", "Agency"]
                    }
                }
                yield f"data: {json.dumps(response_data)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return
            
            # Get intent for AI responses
            intent = await get_intent(query)
            print("Intent: ", intent)
            
            # Determine support message based on intent
            support_message = {}
            stream_generator = None
            
            if intent == "refund":
                support_message = {
                    "label": "would you like to know more about?",
                    "options": ["Rajasthan Patrika", "Refund Policies", "Ad Booking"]
                }
                # Predefined response for refund
                response_text = "To process a refund, please provide your booking details or contact our support team at support@rajasthanpatrika.com. Refunds are typically processed within 5-7 business days after verification."
                yield f"data: {json.dumps({'chunk': response_text})}\n\n"
                final_data = {
                    "message": response_text,
                    "supportMessage": support_message
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return
                
            elif intent == "rp":
                support_message = {
                    "label": "would you like to know more about?",
                    "options": ["Rajasthan Patrika", "Refund Policies", "Ad Booking"]
                }
                # Get RAG context
                try:
                    store = get_vector_store()
                    results = store.query_by_text(query, top_k=5)
                    context_texts = []
                    for match in results.get('matches', []):
                        if 'text' in match['metadata']:
                            context_texts.append(match['metadata']['text'])
                    
                    if context_texts:
                        stream_generator = rag_agent_stream(query, context_texts)
                    else:
                        stream_generator = other_agent_stream(query)
                except Exception as e:
                    print(f"RAG error: {e}")
                    stream_generator = other_agent_stream(query)
                    
            elif intent == "ad_booking":
                support_message = {
                    "label": "For ad booking, please specify the category you are interested in:",
                    "options": ["Customer", "Agency"]
                }
                # Predefined response
                response_text = "To help you with ad booking, could you please let me know which category you want to book an ad for? (Customer, Agency)"
                yield f"data: {json.dumps({'chunk': response_text})}\n\n"
                final_data = {
                    "message": response_text,
                    "supportMessage": support_message
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return
                
            elif intent == "customer":
                support_message = {
                    "label": "Would you like to know the ad booking flow for any other category like Agency, or want to know about Rajasthan Patrika or refund policy?",
                    "options": ["Agency", "Rajasthan Patrika", "Refund Policy"]
                }
                stream_generator = customer_agent_stream(query)
                
            elif intent == "agency":
                support_message = {
                    "label": "Would you like to know the ad booking flow for any other category like Customer, or want to know about Rajasthan Patrika or refund policy?",
                    "options": ["Customer", "Rajasthan Patrika", "Refund Policy"]
                }
                stream_generator = agency_agent_stream(query)
                
            else:  # "other" intent
                support_message = {
                    "label": "Would you like to know the ad booking flow for any other category like Customer, Agency, or want to know about Rajasthan Patrika or refund policy?",
                    "options": ["Customer", "Agency", "Rajasthan Patrika", "Refund Policy"]
                }
                stream_generator = other_agent_stream(query)
            
            # Stream the response if we have a generator
            if stream_generator:
                full_response = ""
                for chunk in stream_generator:
                    if chunk:
                        full_response += chunk
                        # Send each chunk as SSE
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Send final complete response with support message
                final_data = {
                    "message": full_response,
                    "supportMessage": support_message
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
        except Exception as e:
            error_data = {"error": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

    



@router.post("/upload-jsonl")
async def upload_jsonl(
    file: UploadFile = File(..., description="Upload a .jsonl file")
):
    """
    Upload a JSONL file and store its data in the vector database.
    
    Expected JSONL format:
    - Each line should be a valid JSON object
    - Objects should contain text content in fields like 'text', 'content', 'description', etc.
    - Optional fields: 'id', 'url', 'title', 'section', 'metadata'
    
    Returns processing statistics and results.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    
    # Validate file extension
    if not file.filename.lower().endswith('.jsonl'):
        raise HTTPException(
            status_code=415, 
            detail="Unsupported file type. Only .jsonl files are allowed."
        )
    
    # Read file content
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file provided.")
        
        # Check file size (limit to 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(content) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size allowed is {max_size // (1024*1024)}MB"
            )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # Process JSONL file
    try:
        processor = get_jsonl_processor()
        results = processor.process_jsonl_file(content, file.filename)
        
        # Format response
        response_data = {
            "message": "JSONL file processed successfully",
            "filename": results["filename"],
            "file_size_bytes": results["file_size_bytes"],
            "total_objects": results["total_objects"],
            "successful_stores": results["successful_stores"],
            "failed_stores": results["failed_stores"],
            "total_chunks": results["total_chunks"],
            "success_rate": f"{results['success_rate']:.2f}%",
            "processing_details": {
                "objects_processed": results["total_objects"],
                "vectors_created": results["successful_stores"],
                "chunks_created": results["total_chunks"],
                "errors": results["failed_stores"]
            }
        }
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process JSONL file: {str(e)}"
        )




# @router.post("/upload")
# async def extract_text(
#     file: UploadFile = File(..., description="Upload a .pdf or .docx file"),
# ):
#     """
#     Upload a PDF or DOCX and get back its extracted text.
#     - PDFs: extracts all pages.
#     - DOCX: extracts paragraph and table text.
#     - Returns JSON response with filename, file_type, text, and stats.
#     """
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="No filename provided.")

#     # Read & size-check
#     try:
#         content = _enforce_size_limit(file)
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

#     # Extension sniffing
#     name_lower = file.filename.lower()
#     if name_lower.endswith(".pdf"):
#         suffix = ".pdf"
#         ftype = "pdf"
#     elif name_lower.endswith(".docx"):
#         suffix = ".docx"
#         ftype = "docx"
#     else:
#         raise HTTPException(status_code=415, detail="Unsupported file type. Only .pdf and .docx are allowed.")

#     # Persist to temp for libraries that want a path
#     temp_path = None
#     try:
#         temp_path = _save_to_temp(content, suffix=suffix)

#         if ftype == "pdf":
#             text, page_count = _extract_pdf_text(temp_path)
#         else:
#             text = _extract_docx_text(temp_path)
#             page_count = None  # not applicable for DOCX

#         text = text or ""  # ensure string
#         char_count = len(text)
        
#         # Store in vector database
#         print(f"Attempting to store text in vector database...")
#         vector_stored = upsert_texts(text, file.filename, ftype)

#         # JSON response with all stats included
#         payload = {
#             "filename": file.filename,
#             "file_type": ftype,
#             "text": text,
#             "size_bytes": len(content),
#             "pages": page_count,
#             "chars": char_count,
#             "vector_stored": vector_stored,
#             "text_size_bytes": len(text.encode('utf-8')),
#         }
#         return JSONResponse(payload)

#     finally:
#         # Clean up temp file
#         if temp_path and os.path.exists(temp_path):
#             try:
#                 os.remove(temp_path)
#             except Exception:
#                 pass


# @router.post("/query")
# async def query_documents(
#     query: str = Query(..., description="Search query"),
#     top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
#     include_llm_response: bool = Query(True, description="Include LLM-generated response")
# ):
#     """
#     Query the vector database for similar documents and optionally generate an LLM response.
#     """
#     try:
#         store = get_vector_store()
#         results = store.query_by_text(query, top_k=top_k)
        
#         # Format results
#         formatted_results = []
#         context_texts = []
        
#         for match in results.get('matches', []):
#             formatted_results.append({
#                 "id": match['id'],
#                 "score": match['score'],
#                 "metadata": match['metadata']
#             })
#             # Collect context for LLM
#             if 'text' in match['metadata']:
#                 context_texts.append(match['metadata']['text'])
        
#         response_data = {
#             "query": query,
#             "results": formatted_results,
#             "total_results": len(formatted_results)
#         }
        
#         # Generate LLM response if requested and we have context
#         if include_llm_response and context_texts:
#             try:
#                 # Create context for LLM
#                 context = "\n\n".join(context_texts)
                
#                 # Create RAG prompt
#                 rag_prompt = f"""Based on the following context from uploaded documents, please answer the question. If the answer cannot be found in the context, please say so.

# Context:
# {context}

# Question: {query}

# Please provide a comprehensive answer based on the context above:"""
                
#                 # Get LLM response
#                 llm_response = get_google_response(rag_prompt)
                
#                 response_data["llm_response"] = llm_response
#                 response_data["context_used"] = len(context_texts)
                
#             except Exception as llm_error:
#                 response_data["llm_error"] = f"Failed to generate LLM response: {str(llm_error)}"
#                 response_data["llm_response"] = None
#         else:
#             response_data["llm_response"] = None
#             if not context_texts:
#                 response_data["llm_note"] = "No context available for LLM response"
        
#         return JSONResponse(response_data)
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")




# @router.post("/ask")
# async def ask_question(
#     question: str = Query(..., description="Question to ask"),
#     top_k: int = Query(3, ge=1, le=10, description="Number of context documents to retrieve")
# ):
#     """
#     Ask a question and get a fast LLM response using RAG.
#     Returns only the answer text for maximum speed.
#     """
#     try:
#         store = get_vector_store()
#         results = store.query_by_text(question, top_k=top_k)
        
#         # Quickly collect context - only get text, no metadata processing
#         context_texts = []
#         for match in results.get('matches', []):
#             if 'text' in match['metadata']:
#                 context_texts.append(match['metadata']['text'])
        
#         if not context_texts:
#             return "I couldn't find any relevant information in the uploaded documents to answer your question."
        
#         # Create minimal context for LLM
#         context = "\n".join(context_texts[:3])  # Limit to first 3 chunks for speed
        
#         # Simplified prompt for faster processing
#         rag_prompt = f"""Context: {context}

# Question: {question}

# Answer based on the context above:"""
        
#         # Get LLM response
#         llm_response = get_google_response(rag_prompt)
        
#         return llm_response
        
#     except Exception as e:
#         return f"Error: {str(e)}"


