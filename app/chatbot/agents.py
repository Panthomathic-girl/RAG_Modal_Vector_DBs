from app.llm import get_google_response, get_google_response_stream
from app.chatbot.contexts import customer_context, agency_context
import re
import json

def extract_json_from_string(response: str) -> dict:
    try:
        # Check if response is empty or None
        if not response or response.strip() == "":
            raise ValueError("Empty response from LLM")
        
        # Remove markdown fences like ```json ... ```
        cleaned = re.sub(r"```[a-zA-Z]*", "", response).replace("```", "").strip()
        
        # If still empty after cleaning, raise error
        if not cleaned:
            raise ValueError("No valid content found in response")
        
        # Parse JSON
        return json.loads(cleaned)
    
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response if it's embedded in text
        json_match = re.search(r'\{[^}]*"intent"[^}]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # If all else fails, return a default intent
        return {"intent": "other"}
    
    except Exception as e:
        # Return default intent instead of raising error
        print(f"JSON parsing error: {e}")
        return {"intent": "other"}

async def get_intent(query: str):
    try:
        prompt = f'''
        query = {query}
        tell me the intent based on the query
        out of following intents:
        - refund
        - rp (want to know related to the platform rajasthan patrika)
        - ad_booking (want to know related to the advertisement booking)
        - customer (want to know related to book ad for customer)
        - agency (want to know related to book ad for agency)
        - other
        
        {{
            "intent": "refund/rp/ad_booking/customer/agency/other"
        }}

        don't give any other text like ```json or ``` and response just give the json format response
        '''
        
        response = get_google_response(prompt)
        print("response", response)
        
        # Check if response is valid
        if not response or response.strip() == "":
            print("Empty response from LLM, using default intent")
            return "other"
        
        parsed_response = extract_json_from_string(response)
        intent = parsed_response.get("intent", "other")
        
        # Validate intent is one of the allowed values
        valid_intents = ["refund", "rp", "ad_booking", "customer", "agency", "other"]
        if intent not in valid_intents:
            print(f"Invalid intent '{intent}', using default")
            return "other"
        
        return intent
        
    except Exception as e:
        print(f"Error in get_intent: {e}")
        return "other"

async def rag_agent(query: str, context_texts: str):
    try:
        # Create context for LLM
        context = "\n\n".join(context_texts)
        
        # Create RAG prompt
        rag_prompt = f"""
        Context:
        {context}

        Question: {query}

        Based on the context provided above, answer the question directly and concisely.
        If the answer cannot be found in the context, say so clearly.
        Be direct and helpful. Do not include any preparation messages, greetings, or status updates.
        Only provide the actual answer or explanation.

        NOTE: Start your response directly with the answer, no preparation text.
        """
        
        # Get LLM response
        llm_response = get_google_response(rag_prompt)
        return llm_response
        
    except Exception as llm_error:
        raise Exception(f"Failed to generate LLM response: {str(llm_error)}")


async def customer_agent(query: str):
    prompt = f'''
    query = {query}
    context = {customer_context}

    Based on the context provided, answer the user's query directly and concisely.
    If the query is not specific, explain how a customer can book an ad.
    Be direct and helpful. Do not include any preparation messages, greetings, or status updates.
    Only provide the actual answer or explanation.

    NOTE: Start your response directly with the answer, no preparation text.
    '''
    response = get_google_response(prompt, temperature=0.2)
    return response

async def agency_agent(query: str):
    prompt = f'''
    query = {query}
    context = {agency_context}
    
    Based on the context provided, answer the user's query directly and concisely.
    If the query is not specific, explain how an agency can book an ad.
    Be direct and helpful. Do not include any preparation messages, greetings, or status updates.
    Only provide the actual answer or explanation.

    NOTE: Start your response directly with the answer, no preparation text.
    '''
    response = get_google_response(prompt, temperature=0.2)
    return response

async def other_agent(query: str):
    prompt = f'''
    query = {query}
    
    Answer the user's query directly and concisely.
    Be helpful and direct. Do not include any preparation messages, greetings, or status updates.
    Only provide the actual answer or explanation.

    NOTE: Start your response directly with the answer, no preparation text.
    '''
    response = get_google_response(prompt, temperature=0.2)
    return response

# Streaming versions of agents
def rag_agent_stream(query: str, context_texts: str):
    """Stream RAG agent response"""
    try:
        # Create context for LLM
        context = "\n\n".join(context_texts)
        
        # Create RAG prompt
        rag_prompt = f"""
        Context:
        {context}

        Question: {query}

        Based on the context provided above, answer the question directly and concisely.
        If the answer cannot be found in the context, say so clearly.
        Be direct and helpful. Do not include any preparation messages, greetings, or status updates.
        Only provide the actual answer or explanation.

        NOTE: Start your response directly with the answer, no preparation text.
        """
        
        # Stream LLM response
        for chunk in get_google_response_stream(rag_prompt):
            yield chunk
        
    except Exception as llm_error:
        yield f"Failed to generate LLM response: {str(llm_error)}"

def customer_agent_stream(query: str):
    """Stream customer agent response"""
    prompt = f'''
    query = {query}
    context = {customer_context}

    Based on the context provided, answer the user's query directly and concisely.
    If the query is not specific, explain how a customer can book an ad.
    Be direct and helpful. Do not include any preparation messages, greetings, or status updates.
    Only provide the actual answer or explanation.

    NOTE: Start your response directly with the answer, no preparation text.
    '''
    for chunk in get_google_response_stream(prompt, temperature=0.2):
        yield chunk

def agency_agent_stream(query: str):
    """Stream agency agent response"""
    prompt = f'''
    query = {query}
    context = {agency_context}
    
    Based on the context provided, answer the user's query directly and concisely.
    If the query is not specific, explain how an agency can book an ad.
    Be direct and helpful. Do not include any preparation messages, greetings, or status updates.
    Only provide the actual answer or explanation.

    NOTE: Start your response directly with the answer, no preparation text.
    '''
    for chunk in get_google_response_stream(prompt, temperature=0.2):
        yield chunk

def other_agent_stream(query: str):
    """Stream other agent response"""
    prompt = f'''
    query = {query}
    
    Answer the user's query directly and concisely.
    Be helpful and direct. Do not include any preparation messages, greetings, or status updates.
    Only provide the actual answer or explanation.

    NOTE: Start your response directly with the answer, no preparation text.
    '''
    for chunk in get_google_response_stream(prompt, temperature=0.2):
        yield chunk