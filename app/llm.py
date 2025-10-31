import google.generativeai as genai
from app.config import Settings
from PIL import Image
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_gemini():
    """Configure the Gemini API with the API key from settings."""
    try:
        settings = Settings()
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {str(e)}")
        raise

def get_gemini_embedding(text: str, model_name: str = "models/text-embedding-004") -> list:
    """Generate embedding for text using Gemini's text-embedding-004 model."""
    try:
        configure_gemini()
        result = genai.embed_content(
            model=model_name,
            content=text
        )
        embedding = result['embedding']
        logger.info(f"Generated embedding for text: {text[:50]}...")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def get_google_response(prompt: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.7) -> str:
    """
    Generate a text response using Google Gemini model.
    """
    try:
        configure_gemini()
        model = genai.GenerativeModel(model_name)
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=2048,
            top_p=0.8,
            top_k=40
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Return the generated text
        return response.text.strip() if response and response.text else "No response generated."
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error while generating response: {str(e)}"

def get_google_response_stream(prompt: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.7):
    """
    Stream Google Gemini response chunk by chunk.
    Yields text chunks as they arrive from the API.
    """
    try:
        configure_gemini()
        model = genai.GenerativeModel(model_name)
        
        # Configure generation parameters for streaming
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=2048,
            top_p=0.8,
            top_k=40
        )
        
        # Generate content with streaming enabled
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True
        )

        # Yield chunks as they arrive
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        logger.error(f"Error streaming response: {str(e)}")
        yield f"Error while generating response: {str(e)}"

def edit_image_with_gemini(image_path: str, prompt: str, output_path: str = "edited_image.png") -> str:
    """
    Edit an image using Google Gemini's image generation model.
    """
    try:
        configure_gemini()
        # Load the input image
        image_input = Image.open(image_path)

        # Generate content (edit the image)
        model = genai.GenerativeModel("imagen-4.0-generate-001")
        resp = model.generate_content([prompt, image_input])

        # Extract image data from response
        parts = [
            p.inline_data.data
            for p in resp.candidates[0].content.parts
            if getattr(p, "inline_data", None)
        ]

        # Save the edited image
        edited_image = Image.open(BytesIO(parts[0]))
        edited_image.save(output_path)

        logger.info(f"Saved edited image to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error editing image: {str(e)}")
        return ""

def get_image_description(image_bytes: bytes) -> str:
    """
    Convert an image to a text description using Google Gemini VLM.
    Args:
        image_bytes: Bytes of the image.
    Returns:
        Text description of the image.
    """
    try:
        configure_gemini()
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Generate description
        prompt = "Describe the content of this image in detail, including any text, objects, and layout."
        response = model.generate_content([prompt, image])
        description = response.text.strip() if response and response.text else ""
        logger.info("Generated image description")
        return description
    except Exception as e:
        logger.error(f"Error generating image description: {str(e)}")
        return ""


# from groq import Groq
# from app.config import Settings
# import google.generativeai as genai
# from PIL import Image
# from io import BytesIO

# genai.configure(api_key=Settings.GOOGLE_API_KEY)

# def get_google_response(prompt: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.7) -> str:

#     try:
#         model = genai.GenerativeModel(model_name)
        
#         # Configure generation parameters
#         generation_config = genai.types.GenerationConfig(
#             temperature=temperature,
#             max_output_tokens=2048,
#             top_p=0.8,
#             top_k=40
#         )
        
#         response = model.generate_content(
#             prompt,
#             generation_config=generation_config
#         )

#         # Return the generated text
#         return response.text.strip() if response and response.text else "No response generated."

#     except Exception as e:
#         return f"Error while generating response: {e}"

# def get_google_response_stream(prompt: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.7):
#     """
#     Stream Google Gemini response chunk by chunk.
#     Yields text chunks as they arrive from the API.
#     """
#     try:
#         model = genai.GenerativeModel(model_name)
        
#         # Configure generation parameters for streaming
#         generation_config = genai.types.GenerationConfig(
#             temperature=temperature,
#             max_output_tokens=2048,
#             top_p=0.8,
#             top_k=40
#         )
        
#         # Generate content with streaming enabled
#         response = model.generate_content(
#             prompt,
#             generation_config=generation_config,
#             stream=True
#         )

#         # Yield chunks as they arrive
#         for chunk in response:
#             if chunk.text:
#                 yield chunk.text

#     except Exception as e:
#         yield f"Error while generating response: {e}"

# def edit_image_with_gemini(image_path: str, prompt: str, output_path: str = "edited_image.png") -> str:

#     # Load the input image
#     image_input = Image.open(image_path)

#     # Generate content (edit the image)
#     model = genai.GenerativeModel("imagen-4.0-generate-001")
#     resp = model.generate_content([prompt, image_input])

#     # Extract image data from response
#     parts = [
#         p.inline_data.data
#         for p in resp.candidates[0].content.parts
#         if getattr(p, "inline_data", None)
#     ]

#     # Save the edited image
#     edited_image = Image.open(BytesIO(parts[0]))
#     edited_image.save(output_path)

#     print(f"Saved {output_path}")
#     return output_path



